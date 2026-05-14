from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from keyframe_pipeline.config import VisualizationConfig
from keyframe_pipeline.visualizers.plotly_latent import (
    LatentVisualizationStrategy,
    ensure_plotly_available,
    project_latents,
)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cluster_ranges(precluster_summary: dict[str, object] | None, candidate_count: int) -> list[tuple[int, int]]:
    if candidate_count <= 0:
        return []
    raw_ranges = (precluster_summary or {}).get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        return [(0, candidate_count - 1)]

    ranges: list[tuple[int, int]] = []
    for raw_range in raw_ranges:
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            continue
        start = max(0, int(raw_range[0]))
        end = min(candidate_count - 1, int(raw_range[1]))
        if start <= end:
            ranges.append((start, end))
    return ranges or [(0, candidate_count - 1)]


def _candidate_cluster_ids(ranges: list[tuple[int, int]], candidate_count: int) -> np.ndarray:
    cluster_ids = np.zeros(candidate_count, dtype=np.int32)
    for cluster_id, (start, end) in enumerate(ranges):
        cluster_ids[start : end + 1] = cluster_id
    return cluster_ids


def _distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=np.float32)
    return np.linalg.norm(np.diff(points, axis=0), axis=1).astype(np.float32)


def _distance_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "count": int(len(values)),
        "sum": float(np.sum(values)),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean > 1e-12 else 0.0,
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _format_metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _stats_table(title: str, rows: list[tuple[str, float | int]]) -> str:
    body = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(_format_metric(value))}</td></tr>"
        for label, value in rows
    )
    return f"<section class=\"stats-card\"><h2>{html.escape(title)}</h2><table>{body}</table></section>"


def _build_stats_html(
    latents: np.ndarray,
    selected_candidate_orders: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    ranges: list[tuple[int, int]],
) -> str:
    selected_set = set(int(index) for index in selected_candidate_orders.tolist())
    selected_points = latents[selected_candidate_orders] if len(selected_candidate_orders) else np.empty((0, latents.shape[1]))
    path_stats = _distance_stats(_distances(latents))
    selected_stats = _distance_stats(_distances(selected_points))

    intra_selected_distances: list[np.ndarray] = []
    cluster_rows: list[str] = []
    for cluster_id, (start, end) in enumerate(ranges):
        cluster_points = latents[start : end + 1]
        cluster_selected = np.asarray(
            [order for order in selected_candidate_orders if start <= int(order) <= end],
            dtype=np.int32,
        )
        cluster_path_stats = _distance_stats(_distances(cluster_points))
        cluster_selected_stats = _distance_stats(
            _distances(latents[cluster_selected]) if len(cluster_selected) else np.asarray([], dtype=np.float32)
        )
        if len(cluster_selected) >= 2:
            intra_selected_distances.append(_distances(latents[cluster_selected]))
        cluster_rows.append(
            "<tr>"
            f"<td>{cluster_id + 1}</td>"
            f"<td>{start}-{end}</td>"
            f"<td>{end - start + 1}</td>"
            f"<td>{len(cluster_selected)}</td>"
            f"<td>{_format_metric(float(cluster_path_stats['sum']))}</td>"
            f"<td>{_format_metric(float(cluster_selected_stats['mean']))}</td>"
            f"<td>{_format_metric(float(cluster_selected_stats['cv']))}</td>"
            "</tr>"
        )

    intra_stats = _distance_stats(
        np.concatenate(intra_selected_distances) if intra_selected_distances else np.asarray([], dtype=np.float32)
    )
    timestamp_start = float(timestamps_sec[0]) if len(timestamps_sec) else 0.0
    timestamp_end = float(timestamps_sec[-1]) if len(timestamps_sec) else 0.0

    summary = _stats_table(
        "Trajectory",
        [
            ("candidate frames", len(latents)),
            ("selected frames", len(selected_set)),
            ("clusters", len(ranges)),
            ("frame index start", int(frame_indices[0]) if len(frame_indices) else 0),
            ("frame index end", int(frame_indices[-1]) if len(frame_indices) else 0),
            ("time start sec", timestamp_start),
            ("time end sec", timestamp_end),
            ("path length", float(path_stats["sum"])),
            ("adjacent mean", float(path_stats["mean"])),
            ("adjacent std", float(path_stats["std"])),
            ("adjacent CV", float(path_stats["cv"])),
            ("adjacent p10", float(path_stats["p10"])),
            ("adjacent p90", float(path_stats["p90"])),
        ],
    )
    selected = _stats_table(
        "Selected Frames",
        [
            ("distance count", int(selected_stats["count"])),
            ("distance mean", float(selected_stats["mean"])),
            ("distance std", float(selected_stats["std"])),
            ("distance CV", float(selected_stats["cv"])),
            ("distance p10", float(selected_stats["p10"])),
            ("distance p90", float(selected_stats["p90"])),
            ("intra-cluster count", int(intra_stats["count"])),
            ("intra-cluster mean", float(intra_stats["mean"])),
            ("intra-cluster CV", float(intra_stats["cv"])),
        ],
    )
    cluster_table = (
        "<section class=\"stats-card stats-wide\"><h2>Clusters</h2>"
        "<table><thead><tr>"
        "<th>cluster</th><th>candidate order range</th><th>candidate</th><th>selected</th>"
        "<th>path length</th><th>selected mean</th><th>selected CV</th>"
        "</tr></thead><tbody>"
        + "\n".join(cluster_rows)
        + "</tbody></table></section>"
    )
    return f"<div class=\"stats-grid\">{summary}{selected}{cluster_table}</div>"


def _checkbox_html(input_id: str, label: str, checked: bool, data_attrs: dict[str, str]) -> str:
    attrs = " ".join(f"data-{html.escape(key)}=\"{html.escape(value)}\"" for key, value in data_attrs.items())
    checked_attr = " checked" if checked else ""
    return (
        f"<label class=\"control-check\"><input id=\"{html.escape(input_id)}\" "
        f"type=\"checkbox\" {attrs}{checked_attr}> {html.escape(label)}</label>"
    )


class PlotlyLatentControlsVisualizationStrategy(LatentVisualizationStrategy):
    name = "plotly_latent_controls"

    def save(
        self,
        output_path: Path,
        latents: np.ndarray,
        frame_indices: np.ndarray,
        timestamps_sec: np.ndarray,
        selected_candidate_orders: np.ndarray,
        config: VisualizationConfig,
        precluster_summary: dict[str, object] | None = None,
    ) -> None:
        go, pio = ensure_plotly_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        kwargs = config.kwargs or {}
        show_candidate_frames = _as_bool(kwargs.get("show_candidate_frames"), config.show_all_candidates)
        show_selected_frames = _as_bool(kwargs.get("show_selected_frames"), config.show_selected_path)
        show_selected_path = _as_bool(kwargs.get("show_selected_path"), config.show_selected_path)
        show_selected_labels = _as_bool(kwargs.get("show_selected_labels"), config.annotate_every > 0)
        show_frame_path = _as_bool(kwargs.get("show_frame_path"), True)
        show_stats_panel = _as_bool(kwargs.get("show_stats_panel"), True)
        candidate_marker_size = _as_float(kwargs.get("candidate_marker_size"), 2.5 if config.dimensions == 3 else 5.0)
        selected_marker_size = _as_float(kwargs.get("selected_marker_size"), 6.5 if config.dimensions == 3 else 8.0)

        coords = project_latents(latents, config.dimensions)
        ranges = _cluster_ranges(precluster_summary, len(latents))
        candidate_orders = np.arange(len(latents), dtype=np.int32)
        candidate_cluster_ids = _candidate_cluster_ids(ranges, len(latents))
        selected_mask = np.zeros(len(latents), dtype=bool)
        selected_mask[selected_candidate_orders] = True
        custom_data = np.stack(
            [
                candidate_orders,
                frame_indices,
                timestamps_sec,
                selected_mask.astype(np.int32),
                candidate_cluster_ids + 1,
            ],
            axis=1,
        )

        traces: list[object] = []
        trace_meta: list[dict[str, object]] = []

        def add_trace(trace: object, cluster_id: int, kind: str, has_marker: bool) -> None:
            traces.append(trace)
            trace_meta.append(
                {
                    "cluster": str(cluster_id),
                    "kind": kind,
                    "hasMarker": has_marker,
                    "defaultVisible": True,
                }
            )

        colors = [
            "#2563eb",
            "#dc2626",
            "#16a34a",
            "#9333ea",
            "#ea580c",
            "#0891b2",
            "#be123c",
            "#4d7c0f",
        ]
        selected_color = "#f97316"
        candidate_color_max = max(1, len(latents) - 1)

        for cluster_id, (start, end) in enumerate(ranges):
            cluster_slice = slice(start, end + 1)
            cluster_label = f"cluster {cluster_id + 1}"
            color = colors[cluster_id % len(colors)]
            cluster_selected = selected_candidate_orders[
                (selected_candidate_orders >= start) & (selected_candidate_orders <= end)
            ]

            if config.dimensions == 3:
                add_trace(
                    go.Scatter3d(
                        x=coords[cluster_slice, 0],
                        y=coords[cluster_slice, 1],
                        z=coords[cluster_slice, 2],
                        mode="lines",
                        name=f"{cluster_label} frame path",
                        line={"color": color, "width": 4},
                        opacity=0.65,
                        customdata=custom_data[cluster_slice],
                        hovertemplate=(
                            "cluster=%{customdata[4]}<br>candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s"
                            "<br>z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                        ),
                        visible=True if show_frame_path else "legendonly",
                    ),
                    cluster_id,
                    "path",
                    False,
                )
                add_trace(
                    go.Scatter3d(
                        x=[coords[start, 0], coords[end, 0]],
                        y=[coords[start, 1], coords[end, 1]],
                        z=[coords[start, 2], coords[end, 2]],
                        mode="markers+text",
                        name=f"{cluster_label} endpoints",
                        marker={
                            "size": [11, 13],
                            "color": ["#22c55e", "#ef4444"],
                            "symbol": ["diamond", "x"],
                            "line": {"color": "#111827", "width": 2},
                        },
                        text=["start", "end"],
                        textposition=["bottom center", "top center"],
                        customdata=custom_data[[start, end]],
                        hovertemplate=(
                            "cluster=%{customdata[4]} %{text}<br>candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s"
                            "<br>z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                        ),
                        visible=True,
                    ),
                    cluster_id,
                    "endpoint",
                    False,
                )
                add_trace(
                    go.Scatter3d(
                        x=coords[cluster_slice, 0],
                        y=coords[cluster_slice, 1],
                        z=coords[cluster_slice, 2],
                        mode="markers",
                        name=f"{cluster_label} candidate frames",
                        marker={
                            "size": candidate_marker_size,
                            "color": candidate_orders[cluster_slice],
                            "colorscale": "Viridis",
                            "cmin": 0,
                            "cmax": candidate_color_max,
                            "opacity": 0.45,
                            "showscale": cluster_id == 0,
                            **({"colorbar": {"title": "Candidate Order"}} if cluster_id == 0 else {}),
                        },
                        customdata=custom_data[cluster_slice],
                        hovertemplate=(
                            "cluster=%{customdata[4]}<br>candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s<br>"
                            "selected=%{customdata[3]}<br>z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                        ),
                        visible=True if show_candidate_frames else "legendonly",
                    ),
                    cluster_id,
                    "candidate",
                    True,
                )
                if len(cluster_selected):
                    add_trace(
                        go.Scatter3d(
                            x=coords[cluster_selected, 0],
                            y=coords[cluster_selected, 1],
                            z=coords[cluster_selected, 2],
                            mode="lines" if not show_selected_frames else "lines+markers",
                            name=f"{cluster_label} selected frames",
                            line={"color": selected_color, "width": 6},
                            marker={
                                "size": selected_marker_size,
                                "color": selected_color,
                                "line": {"color": "#7f1d1d", "width": 2},
                            },
                            customdata=custom_data[cluster_selected],
                            hovertemplate=(
                                "cluster=%{customdata[4]}<br>candidate_order=%{customdata[0]}<br>"
                                "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s"
                                "<br>z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                            ),
                            visible=True if show_selected_path or show_selected_frames else "legendonly",
                        ),
                        cluster_id,
                        "selected",
                        True,
                    )
                    if show_selected_labels and config.annotate_every > 0:
                        annotated = cluster_selected[:: config.annotate_every]
                        add_trace(
                            go.Scatter3d(
                                x=coords[annotated, 0],
                                y=coords[annotated, 1],
                                z=coords[annotated, 2],
                                mode="text",
                                name=f"{cluster_label} selected labels",
                                text=[str(frame_indices[index]) for index in annotated],
                                textposition="top center",
                                hoverinfo="skip",
                                visible=True if show_selected_frames else "legendonly",
                            ),
                            cluster_id,
                            "selected",
                            False,
                        )
            else:
                add_trace(
                    go.Scatter(
                        x=coords[cluster_slice, 0],
                        y=coords[cluster_slice, 1],
                        mode="lines",
                        name=f"{cluster_label} frame path",
                        line={"color": color, "width": 3},
                        opacity=0.65,
                        customdata=custom_data[cluster_slice],
                        hovertemplate=(
                            "cluster=%{customdata[4]}<br>candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s"
                            "<br>z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                        ),
                        visible=True if show_frame_path else "legendonly",
                    ),
                    cluster_id,
                    "path",
                    False,
                )
                add_trace(
                    go.Scatter(
                        x=[coords[start, 0], coords[end, 0]],
                        y=[coords[start, 1], coords[end, 1]],
                        mode="markers+text",
                        name=f"{cluster_label} endpoints",
                        marker={
                            "size": [11, 13],
                            "color": ["#22c55e", "#ef4444"],
                            "symbol": ["diamond", "x"],
                            "line": {"color": "#111827", "width": 2},
                        },
                        text=["start", "end"],
                        textposition=["bottom center", "top center"],
                        customdata=custom_data[[start, end]],
                        hovertemplate=(
                            "cluster=%{customdata[4]} %{text}<br>candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s"
                            "<br>z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                        ),
                        visible=True,
                    ),
                    cluster_id,
                    "endpoint",
                    False,
                )
                add_trace(
                    go.Scatter(
                        x=coords[cluster_slice, 0],
                        y=coords[cluster_slice, 1],
                        mode="markers",
                        name=f"{cluster_label} candidate frames",
                        marker={
                            "size": candidate_marker_size,
                            "color": candidate_orders[cluster_slice],
                            "colorscale": "Viridis",
                            "cmin": 0,
                            "cmax": candidate_color_max,
                            "opacity": 0.45,
                            "showscale": cluster_id == 0,
                            **({"colorbar": {"title": "Candidate Order"}} if cluster_id == 0 else {}),
                        },
                        customdata=custom_data[cluster_slice],
                        hovertemplate=(
                            "cluster=%{customdata[4]}<br>candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s<br>"
                            "selected=%{customdata[3]}<br>z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                        ),
                        visible=True if show_candidate_frames else "legendonly",
                    ),
                    cluster_id,
                    "candidate",
                    True,
                )
                if len(cluster_selected):
                    add_trace(
                        go.Scatter(
                            x=coords[cluster_selected, 0],
                            y=coords[cluster_selected, 1],
                            mode="lines+markers" if show_selected_path else "markers",
                            name=f"{cluster_label} selected frames",
                            line={"color": selected_color, "width": 3},
                            marker={
                                "size": selected_marker_size,
                                "color": selected_color,
                                "line": {"color": "#7f1d1d", "width": 1},
                            },
                            text=[
                                str(frame_indices[index])
                                if show_selected_labels and config.annotate_every > 0 and order % config.annotate_every == 0
                                else ""
                                for order, index in enumerate(cluster_selected)
                            ],
                            textposition="top center",
                            customdata=custom_data[cluster_selected],
                            hovertemplate=(
                                "cluster=%{customdata[4]}<br>candidate_order=%{customdata[0]}<br>"
                                "frame_idx=%{customdata[1]}<br>time=%{customdata[2]:.3f}s"
                                "<br>z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                            ),
                            visible=True if show_selected_path or show_selected_frames else "legendonly",
                        ),
                        cluster_id,
                        "selected",
                        True,
                    )

        fig = go.Figure(data=traces)
        if config.dimensions == 3:
            fig.update_layout(
                title="Latent Space Keyframe Selection",
                template="plotly_white",
                scene={
                    "xaxis_title": "Latent/PCA 1",
                    "yaxis_title": "Latent/PCA 2",
                    "zaxis_title": "Latent/PCA 3",
                    "aspectmode": "data",
                },
                legend={"x": 0.01, "y": 0.99},
                margin={"l": 0, "r": 0, "t": 50, "b": 0},
            )
        else:
            fig.update_layout(
                title="Latent Space Keyframe Selection",
                template="plotly_white",
                xaxis_title="Latent/PCA 1",
                yaxis_title="Latent/PCA 2",
                legend={"x": 0.01, "y": 0.99},
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
            )

        cluster_controls = "\n".join(
            _checkbox_html(
                input_id=f"cluster-{cluster_id}",
                label=f"Cluster {cluster_id + 1} ({start}-{end})",
                checked=True,
                data_attrs={"cluster": str(cluster_id)},
            )
            for cluster_id, (start, end) in enumerate(ranges)
        )
        layer_controls = "\n".join(
            [
                _checkbox_html("layer-candidate", "Candidate frames", show_candidate_frames, {"layer": "candidate"}),
                _checkbox_html("layer-selected", "Selected frames", show_selected_frames, {"layer": "selected"}),
                _checkbox_html("layer-path", "Frame path", show_frame_path, {"layer": "path"}),
            ]
        )
        stats_html = _build_stats_html(latents, selected_candidate_orders, frame_indices, timestamps_sec, ranges)
        controls_state = {
            "traces": trace_meta,
            "layers": {
                "candidate": bool(show_candidate_frames),
                "selected": bool(show_selected_frames or show_selected_path),
                "path": bool(show_frame_path),
            },
            "clusters": {str(cluster_id): True for cluster_id in range(len(ranges))},
        }
        plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False)

        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Latent Space Keyframe Selection</title>
  <style>
    body {{
      margin: 0;
      background: #f8fafc;
      color: #111827;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      min-height: 100vh;
    }}
    .plot-panel {{
      min-width: 0;
      background: #ffffff;
    }}
    .plot-panel .plotly-graph-div {{
      width: 100% !important;
      height: 100vh !important;
    }}
    .side-panel {{
      border-left: 1px solid #e5e7eb;
      background: #f9fafb;
      padding: 16px;
      overflow: auto;
      max-height: 100vh;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 18px;
      font-weight: 700;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 700;
    }}
    .control-section,
    .stats-card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #ffffff;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .control-check {{
      display: block;
      margin: 8px 0;
      font-size: 13px;
      line-height: 1.35;
      cursor: pointer;
    }}
    .slider-row {{
      margin: 12px 0 4px;
      font-size: 13px;
    }}
    .slider-row label {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 6px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th,
    td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 5px 4px;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      color: #374151;
      font-weight: 600;
      text-align: left;
    }}
    .stats-wide {{
      overflow-x: auto;
    }}
    @media (max-width: 900px) {{
      .layout {{
        display: block;
      }}
      .plot-panel .plotly-graph-div {{
        height: 70vh !important;
      }}
      .side-panel {{
        max-height: none;
        border-left: 0;
        border-top: 1px solid #e5e7eb;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <main class="plot-panel">{plot_html}</main>
    <aside class="side-panel">
      <h1>Latent Controls</h1>
      <section class="control-section">
        <h2>Layers</h2>
        {layer_controls}
      </section>
      <section class="control-section">
        <h2>Clusters</h2>
        {cluster_controls}
      </section>
      <section class="control-section">
        <h2>Point Size</h2>
        <div class="slider-row">
          <label for="candidate-size">Candidate <span id="candidate-size-value">{candidate_marker_size:g}</span></label>
          <input id="candidate-size" type="range" min="1" max="14" step="0.5" value="{candidate_marker_size:g}" data-size-kind="candidate">
        </div>
        <div class="slider-row">
          <label for="selected-size">Selected <span id="selected-size-value">{selected_marker_size:g}</span></label>
          <input id="selected-size" type="range" min="2" max="24" step="0.5" value="{selected_marker_size:g}" data-size-kind="selected">
        </div>
      </section>
      {stats_html if show_stats_panel else ""}
    </aside>
  </div>
  <script>
    (function () {{
      const state = {json.dumps(controls_state, ensure_ascii=False)};
      const plot = document.querySelector(".plotly-graph-div");
      if (!plot || !window.Plotly) {{
        return;
      }}

      function updateVisibility() {{
        const visible = state.traces.map((trace) => {{
          const clusterVisible = state.clusters[trace.cluster] !== false;
          const layerVisible = state.layers[trace.kind] !== false;
          return trace.defaultVisible && clusterVisible && layerVisible ? true : "legendonly";
        }});
        Plotly.restyle(plot, {{ visible }});
      }}

      function setMarkerSize(kind, value) {{
        const indexes = [];
        state.traces.forEach((trace, index) => {{
          if (trace.kind === kind && trace.hasMarker) {{
            indexes.push(index);
          }}
        }});
        if (indexes.length > 0) {{
          Plotly.restyle(plot, {{ "marker.size": Number(value) }}, indexes);
        }}
      }}

      document.querySelectorAll("input[data-layer]").forEach((input) => {{
        input.addEventListener("change", () => {{
          state.layers[input.dataset.layer] = input.checked;
          updateVisibility();
        }});
      }});
      document.querySelectorAll("input[data-cluster]").forEach((input) => {{
        input.addEventListener("change", () => {{
          state.clusters[input.dataset.cluster] = input.checked;
          updateVisibility();
        }});
      }});
      document.querySelectorAll("input[data-size-kind]").forEach((input) => {{
        const valueLabel = document.getElementById(`${{input.id}}-value`);
        input.addEventListener("input", () => {{
          if (valueLabel) {{
            valueLabel.textContent = input.value;
          }}
          setMarkerSize(input.dataset.sizeKind, input.value);
        }});
      }});

      updateVisibility();
    }})();
  </script>
</body>
</html>
"""
        output_path.write_text(page, encoding="utf-8")
