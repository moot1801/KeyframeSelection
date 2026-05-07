from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from keyframe_pipeline.clustering import compute_selected_clusters
from keyframe_pipeline.config import VisualizationConfig
from keyframe_pipeline.visualizers.plotly_latent import (
    LatentVisualizationStrategy,
    ensure_plotly_available,
    project_latents,
)


CLUSTER_COLORS = (
    "#dc2626",
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#c026d3",
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


def _cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def _linear_fractions(latents: np.ndarray) -> np.ndarray:
    if len(latents) == 0:
        return np.asarray([], dtype=np.float32)
    if len(latents) == 1:
        return np.asarray([0.0], dtype=np.float32)

    distances = np.linalg.norm(np.diff(latents, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)]).astype(np.float32)
    total = float(cumulative[-1])
    if total <= 0:
        return np.linspace(0.0, 1.0, len(latents), dtype=np.float32)
    return (cumulative / total).astype(np.float32)


def _linearized_coords(coords: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return coords.copy()

    start = coords[0].astype(np.float32)
    direction = (coords[-1] - coords[0]).astype(np.float32)
    if float(np.linalg.norm(direction)) <= 1e-8:
        span = np.ptp(coords, axis=0).astype(np.float32)
        axis = int(np.argmax(span)) if len(span) else 0
        fallback_span = max(float(span[axis]) if len(span) else 0.0, 1.0)
        direction = np.zeros(coords.shape[1], dtype=np.float32)
        direction[axis] = fallback_span
        start = coords.mean(axis=0).astype(np.float32) - (direction * 0.5)

    return (start[None, :] + fractions[:, None] * direction[None, :]).astype(np.float32)


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
        linear_fractions = _linear_fractions(latents)
        linear_coords = _linearized_coords(coords, linear_fractions)
        candidate_orders = np.arange(len(latents), dtype=np.int32)
        selected_orders = selected_candidate_orders.astype(np.int32)
        selected_coords = coords[selected_orders]
        selected_linear_coords = linear_coords[selected_orders]
        selected_frame_indices = frame_indices[selected_orders]
        selected_timestamps_sec = timestamps_sec[selected_orders]

        selected_distances = np.linalg.norm(np.diff(latents[selected_orders], axis=0), axis=1)
        cluster_info = compute_selected_clusters(
            selected_distances.astype(np.float32),
            latents=latents,
            selected_candidate_orders=selected_orders,
        )
        selected_cluster_ids = cluster_info.cluster_ids
        selected_cluster_colors = [_cluster_color(int(cluster_id)) for cluster_id in selected_cluster_ids]
        candidate_cluster_ids = np.full(len(latents), -1, dtype=np.int32)
        candidate_cluster_ids[selected_orders] = selected_cluster_ids
        selected_mask = np.zeros(len(latents), dtype=bool)
        selected_mask[selected_orders] = True
        custom_data = np.stack(
            [
                candidate_orders,
                frame_indices,
                timestamps_sec,
                selected_mask.astype(np.int32),
                linear_fractions,
                candidate_cluster_ids,
            ],
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
        show_cluster_endpoints = _as_bool(kwargs.get("show_cluster_endpoints"), True)
        show_candidate_order = _as_bool(kwargs.get("show_candidate_order"), False)
        show_linear_view = _as_bool(kwargs.get("show_linear_view"), False)
        selected_marker_size = max(
            2,
            min(24, _as_int(kwargs.get("selected_marker_size"), 6 if config.dimensions == 3 else 8)),
        )
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

        cluster_start_positions: list[int] = []
        cluster_end_positions: list[int] = []
        cluster_start_texts: list[str] = []
        cluster_end_texts: list[str] = []
        for cluster_id in range(cluster_info.cluster_count):
            cluster_positions = np.where(selected_cluster_ids == cluster_id)[0]
            if len(cluster_positions) == 0:
                continue
            start_position = int(cluster_positions[0])
            end_position = int(cluster_positions[-1])
            cluster_start_positions.append(start_position)
            if start_position == end_position:
                cluster_start_texts.append(f"C{cluster_id} start/end")
            else:
                cluster_start_texts.append(f"C{cluster_id} start")
                cluster_end_positions.append(end_position)
                cluster_end_texts.append(f"C{cluster_id} end")
        cluster_start_colors = [_cluster_color(int(selected_cluster_ids[position])) for position in cluster_start_positions]
        cluster_end_colors = [_cluster_color(int(selected_cluster_ids[position])) for position in cluster_end_positions]
        cluster_endpoint_count = len(cluster_start_positions) + len(cluster_end_positions)

        traces: list[object] = []
        trace_map: dict[str, list[int]] = {}
        linear_trace_map: dict[str, list[int]] = {}

        def add_trace(name: str, trace: object) -> None:
            trace_map.setdefault(name, []).append(len(traces))
            traces.append(trace)

        def add_linear_trace(name: str, trace: object) -> None:
            linear_trace_map.setdefault(name, []).append(len(traces))
            traces.append(trace)

        def add_cluster_endpoint_traces(
            add_fn: object,
            scatter_class: object,
            endpoint_coords: np.ndarray,
            visible: bool,
            prefix: str,
        ) -> None:
            is_3d = endpoint_coords.shape[1] >= 3

            def add_endpoint_trace(
                positions: list[int],
                labels: list[str],
                colors: list[str],
                role: str,
                line_color: str,
            ) -> None:
                if len(positions) == 0:
                    return
                endpoint_positions = np.asarray(positions, dtype=np.int32)
                trace_kwargs = {
                    "x": endpoint_coords[endpoint_positions, 0],
                    "y": endpoint_coords[endpoint_positions, 1],
                    "mode": "markers+text",
                    "name": f"{prefix} {role}",
                    "visible": visible,
                    "marker": {
                        "size": selected_marker_size + 5,
                        "color": colors,
                        "line": {"color": line_color, "width": 3},
                    },
                    "text": labels,
                    "textfont": {"color": "#0f172a", "size": 12},
                    "textposition": "bottom center" if role == "starts" else "top center",
                    "customdata": selected_data[endpoint_positions],
                    "hovertemplate": (
                        f"{prefix} {role}<br>"
                        "cluster_id=%{customdata[5]}<br>"
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<extra></extra>"
                    ),
                }
                if is_3d:
                    trace_kwargs["z"] = endpoint_coords[endpoint_positions, 2]
                add_fn("cluster_endpoints", scatter_class(**trace_kwargs))

            add_endpoint_trace(cluster_start_positions, cluster_start_texts, cluster_start_colors, "starts", "#22c55e")
            add_endpoint_trace(cluster_end_positions, cluster_end_texts, cluster_end_colors, "ends", "#0f172a")

        def add_search_result_trace(add_fn: object, scatter_class: object, is_3d: bool, prefix: str) -> None:
            trace_kwargs = {
                "x": [],
                "y": [],
                "mode": "markers+text",
                "name": f"{prefix} search result",
                "visible": False,
                "marker": {
                    "size": selected_marker_size + 8,
                    "color": "#facc15",
                    "line": {"color": "#111827", "width": 4},
                },
                "text": [],
                "textfont": {"color": "#111827", "size": 13},
                "textposition": "middle right",
                "hovertemplate": "%{text}<extra></extra>",
            }
            if is_3d:
                trace_kwargs["z"] = []
            add_fn("search_result", scatter_class(**trace_kwargs))

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
            for cluster_id in range(cluster_info.cluster_count):
                cluster_positions = np.where(selected_cluster_ids == cluster_id)[0]
                if len(cluster_positions) < 2:
                    continue
                add_trace(
                    "selected_path",
                    go.Scatter3d(
                        x=selected_coords[cluster_positions, 0],
                        y=selected_coords[cluster_positions, 1],
                        z=selected_coords[cluster_positions, 2],
                        mode="lines",
                        name=f"selected path cluster {cluster_id}",
                        visible=show_selected_path,
                        line={"color": _cluster_color(cluster_id), "width": 6},
                        customdata=selected_data[cluster_positions],
                        hovertemplate=(
                            f"cluster_id={cluster_id}<br>"
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
                        "size": selected_marker_size,
                        "color": selected_cluster_colors,
                        "line": {"color": "#7f1d1d", "width": 2},
                    },
                    customdata=selected_data,
                    hovertemplate=(
                        "selection_order=%{pointNumber}<br>"
                        "cluster_id=%{customdata[5]}<br>"
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                    ),
                ),
            )
            add_cluster_endpoint_traces(
                add_trace,
                go.Scatter3d,
                selected_coords,
                show_cluster_endpoints,
                "cluster",
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
                        f"C{int(candidate_cluster_ids[candidate_order])} S{order}:F{int(frame_indices[candidate_order])}"
                        for order, candidate_order in zip(selected_label_positions, selected_label_orders)
                    ],
                    textfont={
                        "color": [
                            _cluster_color(int(candidate_cluster_ids[candidate_order]))
                            for candidate_order in selected_label_orders
                        ]
                    },
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
            add_search_result_trace(add_trace, go.Scatter3d, True, "latent")
            if len(linear_coords):
                linear_endpoints = linear_coords[[0, -1]]
                add_linear_trace(
                    "linear_view",
                    go.Scatter3d(
                        x=linear_endpoints[:, 0],
                        y=linear_endpoints[:, 1],
                        z=linear_endpoints[:, 2],
                        mode="lines",
                        name="linear start-end",
                        visible=show_linear_view,
                        line={"color": "#334155", "width": 5},
                        hoverinfo="skip",
                    ),
                )
                add_linear_trace(
                    "candidate_frames",
                    go.Scatter3d(
                        x=linear_coords[:, 0],
                        y=linear_coords[:, 1],
                        z=linear_coords[:, 2],
                        mode="markers",
                        name="linear candidate frames",
                        visible=show_linear_view and show_candidate_frames,
                        marker={
                            "size": 2,
                            "color": candidate_orders,
                            "colorscale": "Viridis",
                            "opacity": 0.45,
                            "showscale": False,
                        },
                        customdata=custom_data,
                        hovertemplate=(
                            "linear candidate<br>"
                            "candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>"
                            "time=%{customdata[2]:.3f}s<br>"
                            "linear_position=%{customdata[4]:.4f}<extra></extra>"
                        ),
                    ),
                )
                for cluster_id in range(cluster_info.cluster_count):
                    cluster_positions = np.where(selected_cluster_ids == cluster_id)[0]
                    if len(cluster_positions) < 2:
                        continue
                    add_linear_trace(
                        "selected_path",
                        go.Scatter3d(
                            x=selected_linear_coords[cluster_positions, 0],
                            y=selected_linear_coords[cluster_positions, 1],
                            z=selected_linear_coords[cluster_positions, 2],
                            mode="lines",
                            name=f"linear selected path cluster {cluster_id}",
                            visible=show_linear_view and show_selected_path,
                            line={"color": _cluster_color(cluster_id), "width": 5},
                            customdata=selected_data[cluster_positions],
                            hovertemplate=(
                                "linear selected path<br>"
                                f"cluster_id={cluster_id}<br>"
                                "candidate_order=%{customdata[0]}<br>"
                                "frame_idx=%{customdata[1]}<br>"
                                "time=%{customdata[2]:.3f}s<br>"
                                "linear_position=%{customdata[4]:.4f}<extra></extra>"
                            ),
                        ),
                    )
                add_linear_trace(
                    "selected_frames",
                    go.Scatter3d(
                        x=selected_linear_coords[:, 0],
                        y=selected_linear_coords[:, 1],
                        z=selected_linear_coords[:, 2],
                        mode="markers",
                        name="linear selected frames",
                        visible=show_linear_view and show_selected_frames,
                        marker={
                            "size": selected_marker_size,
                            "color": selected_cluster_colors,
                            "line": {"color": "#7f1d1d", "width": 2},
                        },
                        customdata=selected_data,
                        hovertemplate=(
                            "linear selected frame<br>"
                            "selection_order=%{pointNumber}<br>"
                            "cluster_id=%{customdata[5]}<br>"
                            "candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>"
                            "time=%{customdata[2]:.3f}s<br>"
                            "linear_position=%{customdata[4]:.4f}<extra></extra>"
                        ),
                    ),
                )
                add_cluster_endpoint_traces(
                    add_linear_trace,
                    go.Scatter3d,
                    selected_linear_coords,
                    show_linear_view and show_cluster_endpoints,
                    "linear cluster",
                )
                add_linear_trace(
                    "selected_labels",
                    go.Scatter3d(
                        x=linear_coords[selected_label_orders, 0],
                        y=linear_coords[selected_label_orders, 1],
                        z=linear_coords[selected_label_orders, 2],
                        mode="text",
                        name="linear selected labels",
                        visible=show_linear_view and show_selected_labels,
                        text=[
                            f"C{int(candidate_cluster_ids[candidate_order])} S{order}:F{int(frame_indices[candidate_order])}"
                            for order, candidate_order in zip(selected_label_positions, selected_label_orders)
                        ],
                        textfont={
                            "color": [
                                _cluster_color(int(candidate_cluster_ids[candidate_order]))
                                for candidate_order in selected_label_orders
                            ]
                        },
                        textposition="top center",
                        hoverinfo="skip",
                    ),
                )
                add_linear_trace(
                    "candidate_order",
                    go.Scatter3d(
                        x=linear_coords[candidate_label_orders, 0],
                        y=linear_coords[candidate_label_orders, 1],
                        z=linear_coords[candidate_label_orders, 2],
                        mode="text",
                        name="linear candidate order",
                        visible=show_linear_view and show_candidate_order,
                        text=[str(int(order)) for order in candidate_label_orders],
                        textposition="middle center",
                        hoverinfo="skip",
                    ),
                )
                add_search_result_trace(add_linear_trace, go.Scatter3d, True, "linear")
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
            for cluster_id in range(cluster_info.cluster_count):
                cluster_positions = np.where(selected_cluster_ids == cluster_id)[0]
                if len(cluster_positions) < 2:
                    continue
                add_trace(
                    "selected_path",
                    go.Scatter(
                        x=selected_coords[cluster_positions, 0],
                        y=selected_coords[cluster_positions, 1],
                        mode="lines",
                        name=f"selected path cluster {cluster_id}",
                        visible=show_selected_path,
                        line={"color": _cluster_color(cluster_id), "width": 3},
                        customdata=selected_data[cluster_positions],
                        hovertemplate=(
                            f"cluster_id={cluster_id}<br>"
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
                        "size": selected_marker_size,
                        "color": selected_cluster_colors,
                        "line": {"color": "#7f1d1d", "width": 1},
                    },
                    customdata=selected_data,
                    hovertemplate=(
                        "selection_order=%{pointNumber}<br>"
                        "cluster_id=%{customdata[5]}<br>"
                        "candidate_order=%{customdata[0]}<br>"
                        "frame_idx=%{customdata[1]}<br>"
                        "time=%{customdata[2]:.3f}s<br>"
                        "z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                    ),
                ),
            )
            add_cluster_endpoint_traces(
                add_trace,
                go.Scatter,
                selected_coords,
                show_cluster_endpoints,
                "cluster",
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
                        f"C{int(candidate_cluster_ids[candidate_order])} S{order}:F{int(frame_indices[candidate_order])}"
                        for order, candidate_order in zip(selected_label_positions, selected_label_orders)
                    ],
                    textfont={
                        "color": [
                            _cluster_color(int(candidate_cluster_ids[candidate_order]))
                            for candidate_order in selected_label_orders
                        ]
                    },
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
            add_search_result_trace(add_trace, go.Scatter, False, "latent")
            if len(linear_coords):
                linear_endpoints = linear_coords[[0, -1]]
                add_linear_trace(
                    "linear_view",
                    go.Scatter(
                        x=linear_endpoints[:, 0],
                        y=linear_endpoints[:, 1],
                        mode="lines",
                        name="linear start-end",
                        visible=show_linear_view,
                        line={"color": "#334155", "width": 3},
                        hoverinfo="skip",
                    ),
                )
                add_linear_trace(
                    "candidate_frames",
                    go.Scatter(
                        x=linear_coords[:, 0],
                        y=linear_coords[:, 1],
                        mode="markers",
                        name="linear candidate frames",
                        visible=show_linear_view and show_candidate_frames,
                        marker={
                            "size": 5,
                            "color": candidate_orders,
                            "colorscale": "Viridis",
                            "opacity": 0.45,
                            "showscale": False,
                        },
                        customdata=custom_data,
                        hovertemplate=(
                            "linear candidate<br>"
                            "candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>"
                            "time=%{customdata[2]:.3f}s<br>"
                            "linear_position=%{customdata[4]:.4f}<extra></extra>"
                        ),
                    ),
                )
                for cluster_id in range(cluster_info.cluster_count):
                    cluster_positions = np.where(selected_cluster_ids == cluster_id)[0]
                    if len(cluster_positions) < 2:
                        continue
                    add_linear_trace(
                        "selected_path",
                        go.Scatter(
                            x=selected_linear_coords[cluster_positions, 0],
                            y=selected_linear_coords[cluster_positions, 1],
                            mode="lines",
                            name=f"linear selected path cluster {cluster_id}",
                            visible=show_linear_view and show_selected_path,
                            line={"color": _cluster_color(cluster_id), "width": 3},
                            customdata=selected_data[cluster_positions],
                            hovertemplate=(
                                "linear selected path<br>"
                                f"cluster_id={cluster_id}<br>"
                                "candidate_order=%{customdata[0]}<br>"
                                "frame_idx=%{customdata[1]}<br>"
                                "time=%{customdata[2]:.3f}s<br>"
                                "linear_position=%{customdata[4]:.4f}<extra></extra>"
                            ),
                        ),
                    )
                add_linear_trace(
                    "selected_frames",
                    go.Scatter(
                        x=selected_linear_coords[:, 0],
                        y=selected_linear_coords[:, 1],
                        mode="markers",
                        name="linear selected frames",
                        visible=show_linear_view and show_selected_frames,
                        marker={
                            "size": selected_marker_size,
                            "color": selected_cluster_colors,
                            "line": {"color": "#7f1d1d", "width": 1},
                        },
                        customdata=selected_data,
                        hovertemplate=(
                            "linear selected frame<br>"
                            "selection_order=%{pointNumber}<br>"
                            "cluster_id=%{customdata[5]}<br>"
                            "candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>"
                            "time=%{customdata[2]:.3f}s<br>"
                            "linear_position=%{customdata[4]:.4f}<extra></extra>"
                        ),
                    ),
                )
                add_cluster_endpoint_traces(
                    add_linear_trace,
                    go.Scatter,
                    selected_linear_coords,
                    show_linear_view and show_cluster_endpoints,
                    "linear cluster",
                )
                add_linear_trace(
                    "selected_labels",
                    go.Scatter(
                        x=linear_coords[selected_label_orders, 0],
                        y=linear_coords[selected_label_orders, 1],
                        mode="text",
                        name="linear selected labels",
                        visible=show_linear_view and show_selected_labels,
                        text=[
                            f"C{int(candidate_cluster_ids[candidate_order])} S{order}:F{int(frame_indices[candidate_order])}"
                            for order, candidate_order in zip(selected_label_positions, selected_label_orders)
                        ],
                        textfont={
                            "color": [
                                _cluster_color(int(candidate_cluster_ids[candidate_order]))
                                for candidate_order in selected_label_orders
                            ]
                        },
                        textposition="top center",
                        hoverinfo="skip",
                    ),
                )
                add_linear_trace(
                    "candidate_order",
                    go.Scatter(
                        x=linear_coords[candidate_label_orders, 0],
                        y=linear_coords[candidate_label_orders, 1],
                        mode="text",
                        name="linear candidate order",
                        visible=show_linear_view and show_candidate_order,
                        text=[str(int(order)) for order in candidate_label_orders],
                        textposition="middle center",
                        hoverinfo="skip",
                    ),
                )
                add_search_result_trace(add_linear_trace, go.Scatter, False, "linear")
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
            "selected_marker_size": int(selected_marker_size),
            "selected_distance_count": int(len(selected_distances)),
            "selected_distance": _distance_stats(selected_distances),
            "selected_cluster_method": "robust selected + candidate context",
            "selected_cluster_global_threshold": float(cluster_info.threshold),
            "selected_cluster_local_window": int(cluster_info.local_window),
            "selected_cluster_local_mad_multiplier": float(cluster_info.local_mad_multiplier),
            "selected_cluster_global_percentile": float(cluster_info.global_percentile),
            "selected_cluster_min_distance_ratio": float(cluster_info.min_distance_ratio),
            "selected_cluster_min_cluster_size": int(cluster_info.min_cluster_size),
            "selected_cluster_candidate_context_window": int(cluster_info.candidate_context_window),
            "selected_cluster_candidate_mad_multiplier": float(cluster_info.candidate_mad_multiplier),
            "selected_cluster_candidate_distance_ratio": float(cluster_info.candidate_distance_ratio),
            "selected_cluster_effective_threshold": _distance_stats(cluster_info.effective_thresholds),
            "selected_cluster_candidate_distance_ratio_stats": _distance_stats(
                cluster_info.candidate_distance_ratios
            ),
            "selected_cluster_selected_boundary_count": int(np.sum(cluster_info.selected_boundary_mask)),
            "selected_cluster_candidate_boundary_count": int(np.sum(cluster_info.candidate_boundary_mask)),
            "selected_cluster_boundary_count": int(np.sum(cluster_info.boundary_mask)),
            "selected_cluster_count": int(cluster_info.cluster_count),
            "selected_cluster_endpoint_count": int(cluster_endpoint_count),
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
            "search_candidate_frame_indices": frame_indices.astype(int).tolist(),
            "search_selected_candidate_orders": selected_orders.astype(int).tolist(),
            "search_selected_frame_indices": selected_frame_indices.astype(int).tolist(),
        }
        initial_state = {
            "linear_view": show_linear_view,
            "candidate_frames": show_candidate_frames,
            "selected_path": show_selected_path,
            "selected_frames": show_selected_frames,
            "selected_labels": show_selected_labels,
            "cluster_endpoints": show_cluster_endpoints,
            "candidate_order": show_candidate_order,
        }

        plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False, auto_play=False)
        page_html = _build_page_html(
            plot_html=plot_html,
            trace_map=trace_map,
            linear_trace_map=linear_trace_map,
            stats=stats,
            initial_state=initial_state,
        )
        output_path.write_text(page_html, encoding="utf-8")


def _build_page_html(
    plot_html: str,
    trace_map: dict[str, list[int]],
    linear_trace_map: dict[str, list[int]],
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
    .range-control,
    .search-control {
      display: grid;
      gap: 8px;
      margin: 8px 0;
      font-size: 14px;
    }
    .range-control input,
    .search-control input,
    .search-control select {
      width: 100%;
      box-sizing: border-box;
      font: inherit;
    }
    .search-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 72px 64px;
      gap: 6px;
    }
    .search-row button,
    .search-control > button {
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #0f172a;
      font: inherit;
      cursor: pointer;
      padding: 5px 8px;
    }
    .search-row button:hover,
    .search-control > button:hover {
      background: #eaf1ff;
      border-color: #93b4ef;
    }
    .search-status {
      min-height: 18px;
      color: #475569;
      font-size: 12px;
      line-height: 1.4;
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
      <h2>View</h2>
      <label class="control"><input id="toggle-linear-view" type="checkbox"> Linear latent view</label>

      <h2>Layers</h2>
      <label class="control"><input id="toggle-candidate-frames" type="checkbox"> Candidate frames</label>
      <label class="control"><input id="toggle-selected-path" type="checkbox"> Selected path</label>
      <label class="control"><input id="toggle-selected-frames" type="checkbox"> Selected frames</label>
      <label class="control"><input id="toggle-selected-labels" type="checkbox"> Selected labels</label>
      <label class="control"><input id="toggle-cluster-endpoints" type="checkbox"> Cluster endpoints</label>
      <label class="control"><input id="toggle-candidate-order" type="checkbox"> Candidate order</label>

      <h2>Style</h2>
      <label class="range-control">
        <span>Selected point size <strong id="selected-marker-size-value"></strong></span>
        <input id="selected-marker-size" type="range" min="2" max="24" step="1">
      </label>

      <h2>Find</h2>
      <div class="search-control">
        <div class="search-row">
          <select id="search-kind">
            <option value="candidate_order">Candidate order</option>
            <option value="frame_index">Frame index</option>
            <option value="selection_order">Selection order</option>
          </select>
          <input id="search-value" type="number" min="0" step="1">
          <button id="search-button" type="button">Find</button>
        </div>
        <button id="search-clear-button" type="button">Clear</button>
        <div id="search-status" class="search-status"></div>
      </div>

      <h2>Visible</h2>
      <table class="stats-table">
        <tbody>
          <tr><th>Linear view</th><td id="visible-linear-view"></td></tr>
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
          <tr><th>Selected clusters</th><td id="selected-cluster-count"></td></tr>
          <tr><th>Cluster method</th><td id="selected-cluster-method"></td></tr>
          <tr><th>Local window</th><td id="selected-cluster-local-window"></td></tr>
          <tr><th>MAD multiplier</th><td id="selected-cluster-local-mad-multiplier"></td></tr>
          <tr><th>Global percentile</th><td id="selected-cluster-global-percentile"></td></tr>
          <tr><th>Distance ratio</th><td id="selected-cluster-min-distance-ratio"></td></tr>
          <tr><th>Min cluster size</th><td id="selected-cluster-min-cluster-size"></td></tr>
          <tr><th>Candidate window</th><td id="selected-cluster-candidate-context-window"></td></tr>
          <tr><th>Candidate ratio</th><td id="selected-cluster-candidate-distance-ratio"></td></tr>
          <tr><th>Candidate ratio range</th><td id="selected-cluster-candidate-distance-ratio-range"></td></tr>
          <tr><th>Selected-only candidates</th><td id="selected-cluster-selected-boundary-count"></td></tr>
          <tr><th>Boundary candidates</th><td id="selected-cluster-candidate-boundary-count"></td></tr>
          <tr><th>Final boundaries</th><td id="selected-cluster-boundary-count"></td></tr>
          <tr><th>Global threshold</th><td id="selected-cluster-global-threshold"></td></tr>
          <tr><th>Effective threshold range</th><td id="selected-cluster-effective-threshold-range"></td></tr>
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
    const latentLinearTraceMap = __LINEAR_TRACE_MAP_JSON__;
    const latentStats = __STATS_JSON__;
    const latentInitialState = __INITIAL_STATE_JSON__;
    const latentLayerKeys = [
      "candidate_frames",
      "selected_path",
      "selected_frames",
      "selected_labels",
      "cluster_endpoints",
      "candidate_order"
    ];
    let currentSearchResult = null;

    function getPlot() {
      return document.querySelector(".plotly-graph-div");
    }

    function getControl(key) {
      const ids = {
        linear_view: "toggle-linear-view",
        candidate_frames: "toggle-candidate-frames",
        selected_path: "toggle-selected-path",
        selected_frames: "toggle-selected-frames",
        selected_labels: "toggle-selected-labels",
        cluster_endpoints: "toggle-cluster-endpoints",
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
      const linearView = getControl("linear_view").checked;
      const candidateFrames = getControl("candidate_frames").checked;
      const selectedFrames = getControl("selected_frames").checked;
      const selectedLabels = getControl("selected_labels").checked;
      const clusterEndpoints = getControl("cluster_endpoints").checked;
      const candidateOrder = getControl("candidate_order").checked;

      const visibleFramePoints = (
        (candidateFrames ? latentStats.candidate_count : 0) +
        (selectedFrames ? latentStats.selected_count : 0) +
        (clusterEndpoints ? latentStats.selected_cluster_endpoint_count : 0)
      );

      setText("visible-linear-view", linearView ? "on" : "off");
      setText("visible-frame-points", formatInteger(visibleFramePoints));
      setText(
        "visible-selected-labels",
        formatInteger(selectedLabels ? latentStats.selected_label_count : 0)
      );
      setText(
        "visible-candidate-order-labels",
        formatInteger(candidateOrder ? latentStats.candidate_order_label_count : 0)
      );
    }

    function updateStaticStats() {
      const distance = latentStats.selected_distance;
      setText("candidate-count", formatInteger(latentStats.candidate_count));
      setText("selected-count", formatInteger(latentStats.selected_count));
      setText("selected-cluster-count", formatInteger(latentStats.selected_cluster_count));
      setText("selected-cluster-method", latentStats.selected_cluster_method);
      setText("selected-cluster-local-window", formatInteger(latentStats.selected_cluster_local_window));
      setText("selected-cluster-local-mad-multiplier", formatFloat(latentStats.selected_cluster_local_mad_multiplier));
      setText("selected-cluster-global-percentile", formatFloat(latentStats.selected_cluster_global_percentile));
      setText("selected-cluster-min-distance-ratio", formatFloat(latentStats.selected_cluster_min_distance_ratio));
      setText("selected-cluster-min-cluster-size", formatInteger(latentStats.selected_cluster_min_cluster_size));
      setText("selected-cluster-candidate-context-window", formatInteger(latentStats.selected_cluster_candidate_context_window));
      setText("selected-cluster-candidate-distance-ratio", formatFloat(latentStats.selected_cluster_candidate_distance_ratio));
      setText(
        "selected-cluster-candidate-distance-ratio-range",
        formatRange(
          latentStats.selected_cluster_candidate_distance_ratio_stats.min,
          latentStats.selected_cluster_candidate_distance_ratio_stats.max,
          formatFloat
        )
      );
      setText("selected-cluster-selected-boundary-count", formatInteger(latentStats.selected_cluster_selected_boundary_count));
      setText("selected-cluster-candidate-boundary-count", formatInteger(latentStats.selected_cluster_candidate_boundary_count));
      setText("selected-cluster-boundary-count", formatInteger(latentStats.selected_cluster_boundary_count));
      setText("selected-cluster-global-threshold", formatFloat(latentStats.selected_cluster_global_threshold));
      setText(
        "selected-cluster-effective-threshold-range",
        formatRange(
          latentStats.selected_cluster_effective_threshold.min,
          latentStats.selected_cluster_effective_threshold.max,
          formatFloat
        )
      );
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

    function setTraceIndicesVisible(indices, visible) {
      const plot = getPlot();
      if (!plot || indices === undefined || !window.Plotly) {
        return;
      }
      const traceIndices = Array.isArray(indices) ? indices : [indices];
      if (traceIndices.length === 0) {
        return;
      }
      Plotly.restyle(plot, {visible}, traceIndices);
    }

    function setTraceMarkerSize(indices, size) {
      const plot = getPlot();
      if (!plot || indices === undefined || !window.Plotly) {
        return;
      }
      const traceIndices = Array.isArray(indices) ? indices : [indices];
      if (traceIndices.length === 0) {
        return;
      }
      Plotly.restyle(plot, {"marker.size": Number(size)}, traceIndices);
    }

    function combinedTraceIndices(key) {
      return [
        ...(latentTraceMap[key] || []),
        ...(latentLinearTraceMap[key] || [])
      ];
    }

    function applySelectedMarkerSize() {
      const control = document.getElementById("selected-marker-size");
      if (!control) {
        return;
      }
      const size = Number(control.value);
      setText("selected-marker-size-value", formatInteger(size));
      setTraceMarkerSize(combinedTraceIndices("selected_frames"), size);
      setTraceMarkerSize(combinedTraceIndices("cluster_endpoints"), size + 5);
      setTraceMarkerSize(combinedTraceIndices("search_result"), size + 8);
    }

    function getActiveCandidateTrace() {
      const linearView = getControl("linear_view").checked;
      const map = linearView ? latentLinearTraceMap : latentTraceMap;
      const traceIndex = map.candidate_frames && map.candidate_frames[0];
      const plot = getPlot();
      if (!plot || traceIndex === undefined) {
        return null;
      }
      return plot.data[traceIndex];
    }

    function findCandidateOrder(kind, rawValue) {
      const value = Number(rawValue);
      if (!Number.isInteger(value)) {
        return null;
      }
      if (kind === "candidate_order") {
        return value >= 0 && value < latentStats.candidate_count ? value : null;
      }
      if (kind === "frame_index") {
        const order = latentStats.search_candidate_frame_indices.indexOf(value);
        return order >= 0 ? order : null;
      }
      if (kind === "selection_order") {
        if (value < 0 || value >= latentStats.selected_count) {
          return null;
        }
        return latentStats.search_selected_candidate_orders[value];
      }
      return null;
    }

    function describeSearchResult(candidateOrder) {
      const frameIndex = latentStats.search_candidate_frame_indices[candidateOrder];
      const selectionOrder = latentStats.search_selected_candidate_orders.indexOf(candidateOrder);
      if (selectionOrder >= 0) {
        return `candidate ${candidateOrder}, frame ${frameIndex}, selected ${selectionOrder}`;
      }
      return `candidate ${candidateOrder}, frame ${frameIndex}`;
    }

    function clearSearchResult() {
      currentSearchResult = null;
      setTraceIndicesVisible(latentTraceMap.search_result, false);
      setTraceIndicesVisible(latentLinearTraceMap.search_result, false);
      const status = document.getElementById("search-status");
      if (status) {
        status.textContent = "";
      }
    }

    function renderSearchResult() {
      const plot = getPlot();
      if (!plot || !window.Plotly) {
        return;
      }
      setTraceIndicesVisible(latentTraceMap.search_result, false);
      setTraceIndicesVisible(latentLinearTraceMap.search_result, false);
      if (!currentSearchResult) {
        return;
      }
      const linearView = getControl("linear_view").checked;
      const sourceTrace = getActiveCandidateTrace();
      const targetMap = linearView ? latentLinearTraceMap : latentTraceMap;
      const searchTraceIndices = targetMap.search_result || [];
      if (!sourceTrace || searchTraceIndices.length === 0) {
        return;
      }
      const candidateOrder = currentSearchResult.candidateOrder;
      const x = sourceTrace.x[candidateOrder];
      const y = sourceTrace.y[candidateOrder];
      const update = {
        x: [[x]],
        y: [[y]],
        text: [[currentSearchResult.label]],
        visible: true
      };
      if (sourceTrace.z) {
        update.z = [[sourceTrace.z[candidateOrder]]];
      }
      Plotly.restyle(plot, update, searchTraceIndices);
    }

    function runSearch() {
      const kind = document.getElementById("search-kind").value;
      const value = document.getElementById("search-value").value;
      const candidateOrder = findCandidateOrder(kind, value);
      const status = document.getElementById("search-status");
      if (candidateOrder === null) {
        clearSearchResult();
        if (status) {
          status.textContent = "No matching node";
        }
        return;
      }
      const label = describeSearchResult(candidateOrder);
      currentSearchResult = {candidateOrder, label};
      if (status) {
        status.textContent = label;
      }
      renderSearchResult();
    }

    function applyVisibility() {
      const linearView = getControl("linear_view").checked;
      setTraceIndicesVisible(latentLinearTraceMap.linear_view, linearView);
      latentLayerKeys.forEach((key) => {
        const enabled = getControl(key).checked;
        setTraceIndicesVisible(latentTraceMap[key], !linearView && enabled);
        setTraceIndicesVisible(latentLinearTraceMap[key], linearView && enabled);
      });
      renderSearchResult();
      updateVisibleStats();
    }

    function setupControls() {
      Object.keys(latentInitialState).forEach((key) => {
        const control = getControl(key);
        if (!control) {
          return;
        }
        control.checked = Boolean(latentInitialState[key]);
        control.addEventListener("change", () => {
          applyVisibility();
        });
      });
      const markerSizeControl = document.getElementById("selected-marker-size");
      if (markerSizeControl) {
        markerSizeControl.value = String(latentStats.selected_marker_size);
        markerSizeControl.addEventListener("input", () => {
          applySelectedMarkerSize();
        });
      }
      const searchButton = document.getElementById("search-button");
      const searchValue = document.getElementById("search-value");
      const clearButton = document.getElementById("search-clear-button");
      if (searchButton) {
        searchButton.addEventListener("click", runSearch);
      }
      if (searchValue) {
        searchValue.addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            runSearch();
          }
        });
      }
      if (clearButton) {
        clearButton.addEventListener("click", clearSearchResult);
      }
      updateStaticStats();
      applySelectedMarkerSize();
      applyVisibility();
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
        .replace("__LINEAR_TRACE_MAP_JSON__", json.dumps(linear_trace_map, ensure_ascii=False))
        .replace("__STATS_JSON__", json.dumps(stats, ensure_ascii=False))
        .replace("__INITIAL_STATE_JSON__", json.dumps(initial_state, ensure_ascii=False))
    )
