from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="연속 프레임 간 latent distance를 계산하고 CSV/요약 파일로 저장합니다."
    )
    parser.add_argument(
        "--latents",
        type=Path,
        required=True,
        help="*_latents.npz 경로",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="거리 CSV 저장 경로. 비우면 artifacts/latents 아래에 자동 저장합니다.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="거리 분포 요약 JSON 저장 경로. 비우면 artifacts/latents 아래에 자동 저장합니다.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="거리 분포를 나눌 범주 개수",
    )
    return parser.parse_args()


def load_latents(latent_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not latent_path.exists():
        raise FileNotFoundError(f"latent 파일을 찾을 수 없습니다: {latent_path}")

    with np.load(latent_path) as data:
        latents = data["latents"].astype(np.float32)
        frame_indices = data["frame_indices"].astype(np.int32)
        timestamps_sec = data["timestamps_sec"].astype(np.float32)

    if latents.ndim != 2:
        raise ValueError("latents 배열은 (N, D) 형태여야 합니다.")
    if len(latents) < 2:
        raise ValueError("연속 프레임 거리를 계산하려면 최소 2개 프레임이 필요합니다.")

    return latents, frame_indices, timestamps_sec


def resolve_output_paths(
    latent_path: Path,
    csv_output: Path | None,
    summary_output: Path | None,
) -> tuple[Path, Path]:
    base_name = latent_path.stem[:-8] if latent_path.stem.endswith("_latents") else latent_path.stem
    default_dir = Path("artifacts/latents")

    resolved_csv = csv_output or default_dir / f"{base_name}_distances.csv"
    resolved_summary = summary_output or default_dir / f"{base_name}_distance_summary.json"
    resolved_csv.parent.mkdir(parents=True, exist_ok=True)
    resolved_summary.parent.mkdir(parents=True, exist_ok=True)
    return resolved_csv, resolved_summary


def compute_consecutive_distances(latents: np.ndarray) -> np.ndarray:
    diffs = latents[1:] - latents[:-1]
    return np.linalg.norm(diffs, axis=1)


def write_distance_csv(
    output_path: Path,
    distances: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_order_n",
                "frame_order_n_plus_1",
                "source_frame_idx_n",
                "source_frame_idx_n_plus_1",
                "time_sec_n",
                "time_sec_n_plus_1",
                "latent_distance",
            ]
        )
        for index, distance in enumerate(distances):
            writer.writerow(
                [
                    index,
                    index + 1,
                    int(frame_indices[index]),
                    int(frame_indices[index + 1]),
                    f"{float(timestamps_sec[index]):.6f}",
                    f"{float(timestamps_sec[index + 1]):.6f}",
                    f"{float(distance):.10f}",
                ]
            )


def build_distribution_summary(distances: np.ndarray, num_bins: int, latent_dim: int) -> dict:
    if num_bins <= 0:
        raise ValueError("--num-bins 값은 1 이상이어야 합니다.")

    min_distance = float(distances.min())
    max_distance = float(distances.max())
    mean_distance = float(distances.mean())
    median_distance = float(np.median(distances))
    std_distance = float(distances.std())
    q1, q3 = np.quantile(distances, [0.25, 0.75])

    if np.isclose(min_distance, max_distance):
        edges = np.array([min_distance, max_distance], dtype=np.float64)
        counts = np.array([len(distances)], dtype=np.int32)
    else:
        counts, edges = np.histogram(distances, bins=num_bins)

    bins: list[dict] = []
    total_count = len(distances)
    for index, count in enumerate(counts):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        bins.append(
            {
                "bin_index": int(index + 1),
                "range": [lower, upper],
                "count": int(count),
                "ratio": float(count / total_count),
            }
        )

    return {
        "num_pairs": int(total_count),
        "latent_dim": int(latent_dim),
        "distance_stats": {
            "min": min_distance,
            "max": max_distance,
            "mean": mean_distance,
            "median": median_distance,
            "std": std_distance,
            "q1": float(q1),
            "q3": float(q3),
        },
        "distribution_bins": bins,
    }


def main() -> None:
    args = parse_args()
    latents, frame_indices, timestamps_sec = load_latents(args.latents)
    csv_output, summary_output = resolve_output_paths(
        latent_path=args.latents,
        csv_output=args.csv_output,
        summary_output=args.summary_output,
    )

    distances = compute_consecutive_distances(latents)
    write_distance_csv(
        output_path=csv_output,
        distances=distances,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
    )

    summary = build_distribution_summary(
        distances=distances,
        num_bins=args.num_bins,
        latent_dim=latents.shape[1],
    )
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV 저장 경로: {csv_output}")
    print(f"요약 저장 경로: {summary_output}")
    print(f"거리 pair 수: {len(distances)}")
    print(
        "거리 통계: "
        f"min={summary['distance_stats']['min']:.10f}, "
        f"max={summary['distance_stats']['max']:.10f}, "
        f"mean={summary['distance_stats']['mean']:.10f}, "
        f"median={summary['distance_stats']['median']:.10f}"
    )
    for item in summary["distribution_bins"]:
        lower, upper = item["range"]
        print(
            f"bin_{item['bin_index']}: "
            f"[{lower:.10f}, {upper:.10f}] "
            f"count={item['count']} "
            f"ratio={item['ratio']:.6f}"
        )


if __name__ == "__main__":
    main()
