from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="latent distance CSV를 바탕으로 상위/하위 pair와 거리 시계열 그래프를 생성합니다."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="latent_distance_report.py가 생성한 distance CSV 경로",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="추출할 상위 distance pair 개수",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=10,
        help="추출할 하위 distance pair 개수",
    )
    parser.add_argument(
        "--top-output",
        type=Path,
        default=None,
        help="상위 pair CSV 저장 경로",
    )
    parser.add_argument(
        "--bottom-output",
        type=Path,
        default=None,
        help="하위 pair CSV 저장 경로",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="거리 시계열 PNG 저장 경로",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="상위/하위 pair 요약 JSON 저장 경로",
    )
    return parser.parse_args()


def load_distance_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"거리 CSV를 찾을 수 없습니다: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("거리 CSV에 데이터가 없습니다.")

    parsed_rows: list[dict] = []
    for row in rows:
        parsed_rows.append(
            {
                "frame_order_n": int(row["frame_order_n"]),
                "frame_order_n_plus_1": int(row["frame_order_n_plus_1"]),
                "source_frame_idx_n": int(row["source_frame_idx_n"]),
                "source_frame_idx_n_plus_1": int(row["source_frame_idx_n_plus_1"]),
                "time_sec_n": float(row["time_sec_n"]),
                "time_sec_n_plus_1": float(row["time_sec_n_plus_1"]),
                "latent_distance": float(row["latent_distance"]),
            }
        )
    return parsed_rows


def resolve_output_paths(
    csv_path: Path,
    top_output: Path | None,
    bottom_output: Path | None,
    plot_output: Path | None,
    summary_output: Path | None,
) -> tuple[Path, Path, Path, Path]:
    base_name = csv_path.stem
    top_path = top_output or Path("artifacts/latents") / f"{base_name}_top10.csv"
    bottom_path = bottom_output or Path("artifacts/latents") / f"{base_name}_bottom10.csv"
    plot_path = plot_output or Path("artifacts/plots") / f"{base_name}_timeseries.png"
    summary_path = summary_output or Path("artifacts/latents") / f"{base_name}_extremes_summary.json"

    for path in (top_path, bottom_path, plot_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    return top_path, bottom_path, plot_path, summary_path


def write_rows_csv(output_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "frame_order_n",
        "frame_order_n_plus_1",
        "source_frame_idx_n",
        "source_frame_idx_n_plus_1",
        "time_sec_n",
        "time_sec_n_plus_1",
        "latent_distance",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_timeseries_plot(
    rows: list[dict],
    top_rows: list[dict],
    bottom_rows: list[dict],
    output_path: Path,
    title: str,
) -> None:
    x_values = [row["frame_order_n"] for row in rows]
    y_values = [row["latent_distance"] for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_values, y_values, color="#2563eb", linewidth=1.4, alpha=0.85, label="distance")

    if top_rows:
        ax.scatter(
            [row["frame_order_n"] for row in top_rows],
            [row["latent_distance"] for row in top_rows],
            color="#dc2626",
            s=35,
            zorder=3,
            label="top distances",
        )
    if bottom_rows:
        ax.scatter(
            [row["frame_order_n"] for row in bottom_rows],
            [row["latent_distance"] for row in bottom_rows],
            color="#16a34a",
            s=35,
            zorder=3,
            label="bottom distances",
        )

    ax.set_title(title)
    ax.set_xlabel("Frame Order n")
    ax.set_ylabel("Latent Distance to n+1")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_summary(top_rows: list[dict], bottom_rows: list[dict], total_rows: int) -> dict:
    return {
        "total_pairs": int(total_rows),
        "top_pairs": top_rows,
        "bottom_pairs": bottom_rows,
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.bottom_k <= 0:
        raise ValueError("--top-k와 --bottom-k는 1 이상이어야 합니다.")

    rows = load_distance_rows(args.csv)
    top_path, bottom_path, plot_path, summary_path = resolve_output_paths(
        csv_path=args.csv,
        top_output=args.top_output,
        bottom_output=args.bottom_output,
        plot_output=args.plot_output,
        summary_output=args.summary_output,
    )

    sorted_desc = sorted(rows, key=lambda item: item["latent_distance"], reverse=True)
    sorted_asc = sorted(rows, key=lambda item: item["latent_distance"])
    top_rows = sorted_desc[: min(args.top_k, len(sorted_desc))]
    bottom_rows = sorted_asc[: min(args.bottom_k, len(sorted_asc))]

    write_rows_csv(top_path, top_rows)
    write_rows_csv(bottom_path, bottom_rows)
    save_timeseries_plot(
        rows=rows,
        top_rows=top_rows,
        bottom_rows=bottom_rows,
        output_path=plot_path,
        title=f"Latent Distance Time Series: {args.csv.stem}",
    )

    summary = build_summary(top_rows=top_rows, bottom_rows=bottom_rows, total_rows=len(rows))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"상위 pair CSV: {top_path}")
    print(f"하위 pair CSV: {bottom_path}")
    print(f"거리 시계열 그래프: {plot_path}")
    print(f"요약 JSON: {summary_path}")


if __name__ == "__main__":
    main()
