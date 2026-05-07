from __future__ import annotations

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
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _distance_stats(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {
            "mean": None,
            "variance": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": _finite_or_none(float(np.mean(values))),
        "variance": _finite_or_none(float(np.var(values))),
        "std": _finite_or_none(float(np.std(values))),
        "min": _finite_or_none(float(np.min(values))),
        "max": _finite_or_none(float(np.max(values))),
    }


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
    ) -> None:
        go, pio = ensure_plotly_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        coords = project_latents(latents, config.dimensions)
        candidate_orders = np.arange(len(latents), dtype=np.int32)
        selected_orders = selected_candidate_orders.astype(np.int32)
        selected_coords = coords[selected_orders]
        selected_frame_indices = frame_indices[selected_orders]
        selected_timestamps_sec = timestamps_sec[selected_orders]

        selected_mask = np.zeros(len(latents), dtype=bool)
        selected_mask[selected_orders] = True
        custom_data = np.stack(
            [candidate_orders, frame_indices, timestamps_sec, selected_mask.astype(np.int32)],
            axis=1,
        )
        selected_data = custom_data[selected_orders]

        kwargs = config.kwargs
        show_candidate_frames = _as_bool(
            kwargs.get("show_candidate_frames"),
            config.show_all_candidates,
        )
        show_selected_path = _as_bool(kwargs.get("show_selected_path"), config.show_selected_path)
        show_selected_frames = _as_bool(kwargs.get("show_selected_frames"), True)
        show_selected_labels = _as_bool(kwargs.get("show_selected_labels"), config.annotate_every > 0)
        show_candidate_order = _as_bool(kwargs.get("show_candidate_order"), False)
        candidate_order_every = max(
            1,
            _as_int(kwargs.get("candidate_order_every"), max(1, len(latents) // 80)),
        )

        selected_label_positions = np.arange(len(selected_orders), dtype=np.int32)
        if config.annotate_every > 0:
            selected_label_positions = selected_label_positions[:: config.annotate_every]
        selected_label_orders = selected_orders[selected_label_positions]

        candidate_label_orders = candidate_orders[::candidate_order_every]
        if len(candidate_orders) and candidate_label_orders[-1] != candidate_orders[-1]:
            candidate_label_orders = np.concatenate([candidate_label_orders, candidate_orders[-1:]])

        traces: list[object] = []
        trace_map: dict[str, int] = {}

        def add_trace(name: str, trace: object) -> None:
            trace_map[name] = len(traces)
            traces.append(trace)

        if config.dimensions == 3:
            add_trace(
                "candidate_frames",
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    name="candidate frames",
                    visible=show_candidate_frames,
                    marker={
                        "size": 2,
                        "color": candidate_orders,
                        "colorscale": "Viridis",
                        "opacity": 0.42,
                        "colorbar": {"title": "Candidate Order"},
                    },
                    customdata=custom_data,
                    hovertemplate=(
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "selected=%{customdata[3]}<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                    ),
                ),
            )
            add_trace(
                "selected_path",
                go.Scatter3d(
                    x=selected_coords[:, 0],
                    y=selected_coords[:, 1],
                    z=selected_coords[:, 2],
                    mode="lines",
                    name="selected path",
                    visible=show_selected_path,
                    line={"color": "#dc2626", "width": 6},
                    customdata=selected_data,
                    hovertemplate=(
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                    ),
                ),
            )
            add_trace(
                "selected_frames",
                go.Scatter3d(
                    x=selected_coords[:, 0],
                    y=selected_coords[:, 1],
                    z=selected_coords[:, 2],
                    mode="markers",
                    name="selected frames",
                    visible=show_selected_frames,
                    marker={
                        "size": 6,
                        "color": "#f97316",
                        "line": {"color": "#7f1d1d", "width": 2},
                    },
                    customdata=selected_data,
                    hovertemplate=(
                        "selection_order=%{pointNumber}<br>"
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                    ),
                ),
            )
            add_trace(
                "selected_labels",
                go.Scatter3d(
                    x=coords[selected_label_orders, 0],
                    y=coords[selected_label_orders, 1],
                    z=coords[selected_label_orders, 2],
                    mode="text",
                    name="selected labels",
                    visible=show_selected_labels,
                    text=[
                        f"S{order}:F{int(frame_indices[candidate_order])}"
                        for order, candidate_order in zip(selected_label_positions, selected_label_orders)
                    ],
                    textposition="top center",
                    hoverinfo="skip",
                ),
            )
            add_trace(
                "candidate_order",
                go.Scatter3d(
                    x=coords[candidate_label_orders, 0],
                    y=coords[candidate_label_orders, 1],
                    z=coords[candidate_label_orders, 2],
                    mode="text",
                    name="candidate order",
                    visible=show_candidate_order,
                    text=[str(int(order)) for order in candidate_label_orders],
                    textposition="middle center",
                    hoverinfo="skip",
                ),
            )
            fig = go.Figure(data=traces)
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
            add_trace(
                "candidate_frames",
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    name="candidate frames",
                    visible=show_candidate_frames,
                    marker={
                        "size": 5,
                        "color": candidate_orders,
                        "colorscale": "Viridis",
                        "opacity": 0.42,
                        "colorbar": {"title": "Candidate Order"},
                    },
                    customdata=custom_data,
                    hovertemplate=(
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "selected=%{customdata[3]}<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                    ),
                ),
            )
            add_trace(
                "selected_path",
                go.Scatter(
                    x=selected_coords[:, 0],
                    y=selected_coords[:, 1],
                    mode="lines",
                    name="selected path",
                    visible=show_selected_path,
                    line={"color": "#dc2626", "width": 3},
                    customdata=selected_data,
                    hovertemplate=(
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                    ),
                ),
            )
            add_trace(
                "selected_frames",
                go.Scatter(
                    x=selected_coords[:, 0],
                    y=selected_coords[:, 1],
                    mode="markers",
                    name="selected frames",
                    visible=show_selected_frames,
                    marker={
                        "size": 8,
                        "color": "#f97316",
                        "line": {"color": "#7f1d1d", "width": 1},
                    },
                    customdata=selected_data,
                    hovertemplate=(
                        "selection_order=%{pointNumber}<br>"
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                    ),
                ),
            )
            add_trace(
                "selected_labels",
                go.Scatter(
                    x=coords[selected_label_orders, 0],
                    y=coords[selected_label_orders, 1],
                    mode="text",
                    name="selected labels",
                    visible=show_selected_labels,
                    text=[
                        f"S{order}:F{int(frame_indices[candidate_order])}"
                        for order, candidate_order in zip(selected_label_positions, selected_label_orders)
                    ],
                    textposition="top center",
                    hoverinfo="skip",
                ),
            )
            add_trace(
                "candidate_order",
                go.Scatter(
                    x=coords[candidate_label_orders, 0],
                    y=coords[candidate_label_orders, 1],
                    mode="text",
                    name="candidate order",
                    visible=show_candidate_order,
                    text=[str(int(order)) for order in candidate_label_orders],
                    textposition="middle center",
                    hoverinfo="skip",
                ),
            )
            fig = go.Figure(data=traces)
            fig.update_layout(
                title="Latent Space Keyframe Selection",
                template="plotly_white",
                xaxis_title="Latent/PCA 1",
                yaxis_title="Latent/PCA 2",
                legend={"x": 0.01, "y": 0.99},
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
            )

        selected_distances = np.linalg.norm(np.diff(latents[selected_orders], axis=0), axis=1)
        stats = {
            "candidate_count": int(len(latents)),
            "selected_count": int(len(selected_orders)),
            "candidate_order_label_count": int(len(candidate_label_orders)),
            "selected_label_count": int(len(selected_label_orders)),
            "selected_distance_count": int(len(selected_distances)),
            "selected_distance": _distance_stats(selected_distances),
            "candidate_order_min": int(candidate_orders[0]) if len(candidate_orders) else None,
            "candidate_order_max": int(candidate_orders[-1]) if len(candidate_orders) else None,
            "frame_index_min": int(frame_indices[0]) if len(frame_indices) else None,
            "frame_index_max": int(frame_indices[-1]) if len(frame_indices) else None,
            "selected_frame_index_min": int(np.min(selected_frame_indices)) if len(selected_frame_indices) else None,
            "selected_frame_index_max": int(np.max(selected_frame_indices)) if len(selected_frame_indices) else None,
            "time_min_sec": _finite_or_none(float(timestamps_sec[0])) if len(timestamps_sec) else None,
            "time_max_sec": _finite_or_none(float(timestamps_sec[-1])) if len(timestamps_sec) else None,
            "selected_time_min_sec": _finite_or_none(float(np.min(selected_timestamps_sec)))
            if len(selected_timestamps_sec)
            else None,
            "selected_time_max_sec": _finite_or_none(float(np.max(selected_timestamps_sec)))
            if len(selected_timestamps_sec)
            else None,
        }
        initial_state = {
            "candidate_frames": show_candidate_frames,
            "selected_path": show_selected_path,
            "selected_frames": show_selected_frames,
            "selected_labels": show_selected_labels,
            "candidate_order": show_candidate_order,
        }

        plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False, auto_play=False)
        page_html = _build_page_html(
            plot_html=plot_html,
            trace_map=trace_map,
            stats=stats,
            initial_state=initial_state,
        )
        output_path.write_text(page_html, encoding="utf-8")


def _build_page_html(
    plot_html: str,
    trace_map: dict[str, int],
    stats: dict[str, object],
    initial_state: dict[str, bool],
) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Latent Space Keyframe Selection</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      color: #172033;
      background: #f5f7fb;
    }
    body {
      margin: 0;
      background: #f5f7fb;
    }
    .page {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      min-height: 100vh;
    }
    .plot-panel {
      min-width: 0;
      background: #ffffff;
    }
    .plot-panel .plotly-graph-div {
      width: 100% !important;
      height: 100vh !important;
    }
    .side-panel {
      border-left: 1px solid #d9e0ec;
      background: #f8fafc;
      padding: 18px;
      overflow-y: auto;
    }
    h1 {
      margin: 0 0 16px;
      font-size: 18px;
      line-height: 1.25;
      font-weight: 700;
    }
    h2 {
      margin: 20px 0 10px;
      font-size: 13px;
      line-height: 1.2;
      font-weight: 700;
      text-transform: uppercase;
      color: #475569;
    }
    .control {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 8px 0;
      font-size: 14px;
      line-height: 1.3;
    }
    .control input {
      width: 16px;
      height: 16px;
      accent-color: #2563eb;
    }
    .stats-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #ffffff;
      border: 1px solid #d9e0ec;
    }
    .stats-table th,
    .stats-table td {
      padding: 8px 9px;
      border-bottom: 1px solid #e5eaf2;
      vertical-align: top;
    }
    .stats-table th {
      text-align: left;
      color: #475569;
      font-weight: 600;
      width: 56%;
    }
    .stats-table td {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: #0f172a;
    }
    .stats-table tr:last-child th,
    .stats-table tr:last-child td {
      border-bottom: 0;
    }
    @media (max-width: 960px) {
      .page {
        grid-template-columns: 1fr;
      }
      .plot-panel .plotly-graph-div {
        height: 72vh !important;
      }
      .side-panel {
        border-left: 0;
        border-top: 1px solid #d9e0ec;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="plot-panel">
      __PLOT_HTML__
    </section>
    <aside class="side-panel">
      <h1>Latent Space Controls</h1>
      <h2>Layers</h2>
      <label class="control"><input id="toggle-candidate-frames" type="checkbox"> Candidate frames</label>
      <label class="control"><input id="toggle-selected-path" type="checkbox"> Selected path</label>
      <label class="control"><input id="toggle-selected-frames" type="checkbox"> Selected frames</label>
      <label class="control"><input id="toggle-selected-labels" type="checkbox"> Selected labels</label>
      <label class="control"><input id="toggle-candidate-order" type="checkbox"> Candidate order</label>

      <h2>Visible</h2>
      <table class="stats-table">
        <tbody>
          <tr><th>Visible frame points</th><td id="visible-frame-points"></td></tr>
          <tr><th>Visible selected labels</th><td id="visible-selected-labels"></td></tr>
          <tr><th>Visible candidate order labels</th><td id="visible-candidate-order-labels"></td></tr>
        </tbody>
      </table>

      <h2>Selection</h2>
      <table class="stats-table">
        <tbody>
          <tr><th>Candidate frames</th><td id="candidate-count"></td></tr>
          <tr><th>Selected frames</th><td id="selected-count"></td></tr>
          <tr><th>Distance segments</th><td id="distance-count"></td></tr>
          <tr><th>Distance mean</th><td id="distance-mean"></td></tr>
          <tr><th>Distance variance</th><td id="distance-variance"></td></tr>
          <tr><th>Distance std</th><td id="distance-std"></td></tr>
          <tr><th>Distance min</th><td id="distance-min"></td></tr>
          <tr><th>Distance max</th><td id="distance-max"></td></tr>
        </tbody>
      </table>

      <h2>Ranges</h2>
      <table class="stats-table">
        <tbody>
          <tr><th>Candidate order</th><td id="candidate-order-range"></td></tr>
          <tr><th>Frame index</th><td id="frame-index-range"></td></tr>
          <tr><th>Selected frame index</th><td id="selected-frame-index-range"></td></tr>
          <tr><th>Time span sec</th><td id="time-range"></td></tr>
          <tr><th>Selected time span sec</th><td id="selected-time-range"></td></tr>
        </tbody>
      </table>
    </aside>
  </main>
  <script>
    const latentTraceMap = __TRACE_MAP_JSON__;
    const latentStats = __STATS_JSON__;
    const latentInitialState = __INITIAL_STATE_JSON__;

    function getPlot() {
      return document.querySelector(".plotly-graph-div");
    }

    function getControl(key) {
      const ids = {
        candidate_frames: "toggle-candidate-frames",
        selected_path: "toggle-selected-path",
        selected_frames: "toggle-selected-frames",
        selected_labels: "toggle-selected-labels",
        candidate_order: "toggle-candidate-order"
      };
      return document.getElementById(ids[key]);
    }

    function formatInteger(value) {
      if (value === null || value === undefined) {
        return "n/a";
      }
      return Number(value).toLocaleString();
    }

    function formatFloat(value) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "n/a";
      }
      return Number(value).toFixed(6);
    }

    function formatRange(start, end, formatter) {
      if (start === null || start === undefined || end === null || end === undefined) {
        return "n/a";
      }
      return `${formatter(start)} - ${formatter(end)}`;
    }

    function setText(id, value) {
      const element = document.getElementById(id);
      if (element) {
        element.textContent = value;
      }
    }

    function updateVisibleStats() {
      const candidateFrames = getControl("candidate_frames").checked;
      const selectedFrames = getControl("selected_frames").checked;
      const selectedLabels = getControl("selected_labels").checked;
      const candidateOrder = getControl("candidate_order").checked;

      const visibleFramePoints = candidateFrames
        ? latentStats.candidate_count
        : selectedFrames
          ? latentStats.selected_count
          : 0;

      setText("visible-frame-points", formatInteger(visibleFramePoints));
      setText("visible-selected-labels", formatInteger(selectedLabels ? latentStats.selected_label_count : 0));
      setText(
        "visible-candidate-order-labels",
        formatInteger(candidateOrder ? latentStats.candidate_order_label_count : 0)
      );
    }

    function updateStaticStats() {
      const distance = latentStats.selected_distance;
      setText("candidate-count", formatInteger(latentStats.candidate_count));
      setText("selected-count", formatInteger(latentStats.selected_count));
      setText("distance-count", formatInteger(latentStats.selected_distance_count));
      setText("distance-mean", formatFloat(distance.mean));
      setText("distance-variance", formatFloat(distance.variance));
      setText("distance-std", formatFloat(distance.std));
      setText("distance-min", formatFloat(distance.min));
      setText("distance-max", formatFloat(distance.max));
      setText(
        "candidate-order-range",
        formatRange(latentStats.candidate_order_min, latentStats.candidate_order_max, formatInteger)
      );
      setText(
        "frame-index-range",
        formatRange(latentStats.frame_index_min, latentStats.frame_index_max, formatInteger)
      );
      setText(
        "selected-frame-index-range",
        formatRange(latentStats.selected_frame_index_min, latentStats.selected_frame_index_max, formatInteger)
      );
      setText("time-range", formatRange(latentStats.time_min_sec, latentStats.time_max_sec, formatFloat));
      setText(
        "selected-time-range",
        formatRange(latentStats.selected_time_min_sec, latentStats.selected_time_max_sec, formatFloat)
      );
    }

    function setTraceVisible(key, visible) {
      const plot = getPlot();
      if (!plot || latentTraceMap[key] === undefined || !window.Plotly) {
        return;
      }
      Plotly.restyle(plot, {visible}, [latentTraceMap[key]]);
    }

    function setupControls() {
      Object.keys(latentInitialState).forEach((key) => {
        const control = getControl(key);
        if (!control) {
          return;
        }
        control.checked = Boolean(latentInitialState[key]);
        control.addEventListener("change", () => {
          setTraceVisible(key, control.checked);
          updateVisibleStats();
        });
      });
      updateStaticStats();
      updateVisibleStats();
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", setupControls);
    } else {
      setupControls();
    }
  </script>
</body>
</html>
"""
    return (
        template.replace("__PLOT_HTML__", plot_html)
        .replace("__TRACE_MAP_JSON__", json.dumps(trace_map, ensure_ascii=False))
        .replace("__STATS_JSON__", json.dumps(stats, ensure_ascii=False))
        .replace("__INITIAL_STATE_JSON__", json.dumps(initial_state, ensure_ascii=False))
    )
