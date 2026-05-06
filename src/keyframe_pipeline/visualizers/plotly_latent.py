from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from keyframe_pipeline.config import VisualizationConfig


class LatentVisualizationStrategy(ABC):
    name: str

    @abstractmethod
    def save(
        self,
        output_path: Path,
        latents: np.ndarray,
        frame_indices: np.ndarray,
        timestamps_sec: np.ndarray,
        selected_candidate_orders: np.ndarray,
        config: VisualizationConfig,
    ) -> None:
        raise NotImplementedError


def ensure_plotly_available() -> tuple[object, object]:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise RuntimeError(
            "latent HTML 시각화를 사용하려면 plotly가 필요합니다. requirements.txt를 다시 설치하세요."
        ) from exc
    return go, pio


def project_latents(latents: np.ndarray, dimensions: int) -> np.ndarray:
    if latents.shape[1] == dimensions:
        return latents.astype(np.float32)

    centered = latents - latents.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[: min(dimensions, vt.shape[0])].T
    if projected.shape[1] < dimensions:
        padding = np.zeros((len(projected), dimensions - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected, padding], axis=1)
    return projected.astype(np.float32)


def save_frame_index_comparison_plot(
    output_path: Path,
    selected_frame_indices: np.ndarray,
    uniform_frame_indices: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_order = np.arange(len(selected_frame_indices))
    uniform_order = np.arange(len(uniform_frame_indices))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        uniform_order,
        uniform_frame_indices,
        marker="o",
        linewidth=2,
        markersize=4,
        color="#2563eb",
        label="Uniform sampling",
    )
    ax.plot(
        selected_order,
        selected_frame_indices,
        marker="o",
        linewidth=2,
        markersize=4,
        color="#dc2626",
        label="Keyframe selection",
    )
    ax.set_title("Selected Frame Index Comparison")
    ax.set_xlabel("Selection order")
    ax.set_ylabel("Original video frame index")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


class PlotlyLatentVisualizationStrategy(LatentVisualizationStrategy):
    name = "plotly_latent"

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
        candidate_orders = np.arange(len(latents))
        selected_mask = np.zeros(len(latents), dtype=bool)
        selected_mask[selected_candidate_orders] = True
        custom_data = np.stack(
            [candidate_orders, frame_indices, timestamps_sec, selected_mask.astype(np.int32)],
            axis=1,
        )

        traces: list[object] = []
        if config.dimensions == 3:
            if config.show_all_candidates:
                traces.append(
                    go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode="markers",
                        name="candidate frames",
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
                    )
                )
            if config.show_selected_path:
                selected_coords = coords[selected_candidate_orders]
                selected_data = custom_data[selected_candidate_orders]
                traces.append(
                    go.Scatter3d(
                        x=selected_coords[:, 0],
                        y=selected_coords[:, 1],
                        z=selected_coords[:, 2],
                        mode="lines+markers",
                        name="selected path",
                        line={"color": "#dc2626", "width": 6},
                        marker={
                            "size": 6,
                            "color": "#f97316",
                            "line": {"color": "#7f1d1d", "width": 2},
                        },
                        customdata=selected_data,
                        hovertemplate=(
                            "candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>"
                            "time=%{customdata[2]:.3f}s<br>"
                            "z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
                        ),
                    )
                )

            if config.annotate_every > 0:
                annotated = selected_candidate_orders[:: config.annotate_every]
                traces.append(
                    go.Scatter3d(
                        x=coords[annotated, 0],
                        y=coords[annotated, 1],
                        z=coords[annotated, 2],
                        mode="text",
                        name="selected labels",
                        text=[str(index) for index in frame_indices[annotated]],
                        textposition="top center",
                        hoverinfo="skip",
                    )
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
            if config.show_all_candidates:
                traces.append(
                    go.Scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode="markers",
                        name="candidate frames",
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
                    )
                )
            if config.show_selected_path:
                selected_coords = coords[selected_candidate_orders]
                selected_data = custom_data[selected_candidate_orders]
                traces.append(
                    go.Scatter(
                        x=selected_coords[:, 0],
                        y=selected_coords[:, 1],
                        mode="lines+markers+text" if config.annotate_every > 0 else "lines+markers",
                        name="selected path",
                        line={"color": "#dc2626", "width": 3},
                        marker={
                            "size": 8,
                            "color": "#f97316",
                            "line": {"color": "#7f1d1d", "width": 1},
                        },
                        text=[
                            str(frame_indices[index])
                            if config.annotate_every > 0 and order % config.annotate_every == 0
                            else ""
                            for order, index in enumerate(selected_candidate_orders)
                        ],
                        textposition="top center",
                        customdata=selected_data,
                        hovertemplate=(
                            "candidate_order=%{customdata[0]}<br>"
                            "frame_idx=%{customdata[1]}<br>"
                            "time=%{customdata[2]:.3f}s<br>"
                            "z1=%{x:.4f}<br>z2=%{y:.4f}<extra></extra>"
                        ),
                    )
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

        pio.write_html(fig, file=str(output_path), auto_open=False, include_plotlyjs=True, full_html=True)


class DisabledVisualizationStrategy(LatentVisualizationStrategy):
    name = "none"

    def save(
        self,
        output_path: Path,
        latents: np.ndarray,
        frame_indices: np.ndarray,
        timestamps_sec: np.ndarray,
        selected_candidate_orders: np.ndarray,
        config: VisualizationConfig,
    ) -> None:
        return None


VISUALIZATION_STRATEGIES: dict[str, LatentVisualizationStrategy] = {
    strategy.name: strategy
    for strategy in (
        PlotlyLatentVisualizationStrategy(),
        DisabledVisualizationStrategy(),
    )
}


def build_visualizer(config: VisualizationConfig) -> LatentVisualizationStrategy:
    try:
        return VISUALIZATION_STRATEGIES[config.name]
    except KeyError as exc:
        available = ", ".join(sorted(VISUALIZATION_STRATEGIES))
        raise ValueError(f"지원하지 않는 visualization 전략입니다: {config.name}. available={available}") from exc
