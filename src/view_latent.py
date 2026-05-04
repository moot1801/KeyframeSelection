from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D latent space 상에서 프레임 시간 흐름을 시각화합니다.")
    parser.add_argument(
        "--latents",
        type=Path,
        required=True,
        help="2d_AE.py가 생성한 *_latents.npz 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="시각화 이미지 저장 경로. 비우면 artifacts/plots에 자동 저장합니다.",
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
        help="N 간격으로 프레임 번호를 표기합니다. 0이면 표기하지 않습니다.",
    )
    return parser.parse_args()


def load_latents(latent_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if not latent_path.exists():
        raise FileNotFoundError(f"latent 파일을 찾을 수 없습니다: {latent_path}")

    with np.load(latent_path) as data:
        latents = data["latents"].astype(np.float32)
        frame_indices = data["frame_indices"].astype(np.int32)
        timestamps_sec = data["timestamps_sec"].astype(np.float32)
        run_name = str(data["run_name"].item()) if "run_name" in data else latent_path.stem

    if latents.ndim != 2 or latents.shape[1] != 2:
        raise ValueError("시각화를 위해서는 latent shape가 (N, 2)여야 합니다.")

    return latents, frame_indices, timestamps_sec, run_name


def resolve_output_path(output: Path | None, run_name: str) -> Path:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    default_path = Path("artifacts/plots") / f"{run_name}_trajectory.png"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path


def draw_latent_path(
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    output_path: Path,
    title: str,
    annotate_every: int,
) -> None:
    order = np.arange(len(latents))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(latents[:, 0], latents[:, 1], color="#94a3b8", linewidth=1.5, alpha=0.9)
    scatter = ax.scatter(
        latents[:, 0],
        latents[:, 1],
        c=order,
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.scatter(latents[0, 0], latents[0, 1], color="#16a34a", s=150, marker="o", label="start")
    ax.scatter(latents[-1, 0], latents[-1, 1], color="#dc2626", s=170, marker="X", label="end")

    if annotate_every > 0:
        for index in range(0, len(latents), annotate_every):
            ax.annotate(
                f"{index}",
                (latents[index, 0], latents[index, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Frame Order")

    summary = (
        f"frames={len(latents)} | frame_idx={int(frame_indices[0])}->{int(frame_indices[-1])} "
        f"| time={timestamps_sec[0]:.3f}s->{timestamps_sec[-1]:.3f}s"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    latents, frame_indices, timestamps_sec, run_name = load_latents(args.latents)
    output_path = resolve_output_path(args.output, run_name)
    title = args.title or f"Latent Trajectory: {run_name}"

    draw_latent_path(
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        output_path=output_path,
        title=title,
        annotate_every=args.annotate_every,
    )
    print(f"시각화 저장 경로: {output_path}")


if __name__ == "__main__":
    main()
