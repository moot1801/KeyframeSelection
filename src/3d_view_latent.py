from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D latent space를 인터랙티브 HTML로 시각화합니다.")
    parser.add_argument(
        "--latents",
        type=Path,
        required=True,
        help="3d_AE.py가 생성한 *_3d_latents.npz 또는 epoch_*_latents.npz 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="시각화 HTML 저장 경로. 비우면 artifacts/plots에 자동 저장합니다.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="그래프 제목",
    )
    parser.add_argument(
        "--annotate-every",
        type=int,
        default=0,
        help="N 간격으로 프레임 번호를 텍스트로 표시합니다. 0이면 표시하지 않습니다.",
    )
    return parser.parse_args()


def ensure_plotly_available() -> tuple[object, object]:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise RuntimeError(
            "3D latent 인터랙티브 시각화를 사용하려면 plotly가 필요합니다. requirements.txt를 다시 설치하세요."
        ) from exc
    return go, pio


def load_latents(latent_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if not latent_path.exists():
        raise FileNotFoundError(f"latent 파일을 찾을 수 없습니다: {latent_path}")

    with np.load(latent_path) as data:
        latents = data["latents"].astype(np.float32)
        frame_indices = data["frame_indices"].astype(np.int32)
        timestamps_sec = data["timestamps_sec"].astype(np.float32)

    if latents.ndim != 2 or latents.shape[1] != 3:
        raise ValueError("시각화를 위해서는 latent shape가 (N, 3)이어야 합니다.")

    base_name = latent_path.stem[:-8] if latent_path.stem.endswith("_latents") else latent_path.stem
    return latents, frame_indices, timestamps_sec, base_name


def resolve_output_path(latent_path: Path, output: Path | None) -> Path:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    base_name = latent_path.stem[:-8] if latent_path.stem.endswith("_latents") else latent_path.stem
    default_path = Path("artifacts/plots") / f"{base_name}_trajectory.html"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path


def build_interactive_figure(
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    title: str,
    annotate_every: int,
) -> object:
    go, _ = ensure_plotly_available()
    order = np.arange(len(latents))
    custom_data = np.stack([order, frame_indices, timestamps_sec], axis=1)
    start_data = np.array([[order[0], frame_indices[0], timestamps_sec[0]]], dtype=np.float32)
    end_data = np.array([[order[-1], frame_indices[-1], timestamps_sec[-1]]], dtype=np.float32)

    traces: list[object] = [
        go.Scatter3d(
            x=latents[:, 0],
            y=latents[:, 1],
            z=latents[:, 2],
            mode="lines",
            name="trajectory-shadow",
            line={"color": "rgba(15, 23, 42, 0.18)", "width": 10},
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter3d(
            x=latents[:, 0],
            y=latents[:, 1],
            z=latents[:, 2],
            mode="lines",
            name="trajectory",
            line={"color": "rgba(37, 99, 235, 0.96)", "width": 5},
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=latents[:, 0],
            y=latents[:, 1],
            z=latents[:, 2],
            mode="markers",
            name="frames",
            marker={
                "size": 2,
                "color": order,
                "colorscale": "Viridis",
                "opacity": 0.58,
                "colorbar": {"title": "Frame Order"},
            },
            customdata=custom_data,
            hovertemplate=(
                "order=%{customdata[0]}<br>"
                "frame_idx=%{customdata[1]}<br>"
                "time=%{customdata[2]:.3f}s<br>"
                "z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
            ),
        ),
        go.Scatter3d(
            x=[latents[0, 0]],
            y=[latents[0, 1]],
            z=[latents[0, 2]],
            mode="markers+text",
            name="start",
            marker={
                "size": 8,
                "color": "#16a34a",
                "symbol": "circle",
                "line": {"color": "#052e16", "width": 5},
            },
            text=["START"],
            textposition="top center",
            textfont={"size": 12, "color": "#166534"},
            customdata=start_data,
            hovertemplate=(
                "start<br>"
                "order=%{customdata[0]}<br>"
                "frame_idx=%{customdata[1]}<br>"
                "time=%{customdata[2]:.3f}s<extra></extra>"
            ),
        ),
        go.Scatter3d(
            x=[latents[-1, 0]],
            y=[latents[-1, 1]],
            z=[latents[-1, 2]],
            mode="markers+text",
            name="end",
            marker={
                "size": 9,
                "color": "#dc2626",
                "symbol": "diamond",
                "line": {"color": "#7f1d1d", "width": 5},
            },
            text=["END"],
            textposition="top center",
            textfont={"size": 12, "color": "#991b1b"},
            customdata=end_data,
            hovertemplate=(
                "end<br>"
                "order=%{customdata[0]}<br>"
                "frame_idx=%{customdata[1]}<br>"
                "time=%{customdata[2]:.3f}s<extra></extra>"
            ),
        ),
    ]

    if annotate_every > 0:
        annotated_indices = np.arange(0, len(latents), annotate_every)
        traces.append(
            go.Scatter3d(
                x=latents[annotated_indices, 0],
                y=latents[annotated_indices, 1],
                z=latents[annotated_indices, 2],
                mode="text",
                name="labels",
                text=[str(index) for index in annotated_indices],
                textposition="top center",
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        template="plotly_white",
        scene={
            "xaxis_title": "Latent Dimension 1",
            "yaxis_title": "Latent Dimension 2",
            "zaxis_title": "Latent Dimension 3",
            "aspectmode": "data",
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def save_interactive_plot(
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    output_path: Path,
    title: str,
    annotate_every: int,
) -> None:
    _, pio = ensure_plotly_available()
    fig = build_interactive_figure(
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        title=title,
        annotate_every=annotate_every,
    )
    pio.write_html(fig, file=str(output_path), auto_open=False, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    if args.annotate_every < 0:
        raise ValueError("--annotate-every 값은 0 이상이어야 합니다.")

    latents, frame_indices, timestamps_sec, base_name = load_latents(args.latents)
    output_path = resolve_output_path(args.latents, args.output)
    title = args.title or f"3D Latent Trajectory: {base_name}"

    save_interactive_plot(
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        output_path=output_path,
        title=title,
        annotate_every=args.annotate_every,
    )
    print(f"인터랙티브 시각화 저장 경로: {output_path}")


if __name__ == "__main__":
    main()
