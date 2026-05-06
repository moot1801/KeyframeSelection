from __future__ import annotations

import argparse
from pathlib import Path

from keyframe_pipeline.config import PipelineConfig, parse_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "영상 후보 프레임으로 autoencoder를 학습한 뒤 latent 거리 분산이 낮은 "
            "실제 영상 프레임 집합을 선택합니다."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML 설정 파일 경로")
    return parser.parse_args()


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    import numpy as np

    from keyframe_pipeline.models import build_model
    from keyframe_pipeline.outputs import (
        save_checkpoint,
        save_latent_npz,
        save_metrics_json,
        write_selected_csv,
    )
    from keyframe_pipeline.selectors import build_selector, validate_selection_result
    from keyframe_pipeline.trainer import train_autoencoder
    from keyframe_pipeline.utils import resolve_device, set_seed
    from keyframe_pipeline.video import (
        compute_sample_indices,
        encode_video_frames,
        export_selected_images,
        extract_candidate_frames,
    )
    from keyframe_pipeline.visualizers import build_visualizer, save_frame_index_comparison_plot

    set_seed(config.train.seed)

    output_dir = config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] 학습용 후보 프레임 추출")
    training_batch = extract_candidate_frames(
        video_path=config.video.input_path,
        candidate_num_frames=config.video.candidate_num_frames,
        image_size=config.video.image_size,
        color_mode=config.video.color_mode,
    )

    print("[2/8] autoencoder 학습")
    device = resolve_device(config.train.device)
    model = build_model(
        config=config.model,
        in_channels=training_batch.frames.shape[1],
        input_height=training_batch.frames.shape[-2],
        input_width=training_batch.frames.shape[-1],
    )
    history = train_autoencoder(
        model=model,
        frames=training_batch.frames,
        train_config=config.train,
        optimizer_config=config.optimizer,
        loss_config=config.loss,
        device=device,
    )

    print("[3/8] 선택용 전체 영상 프레임 latent 추출")
    latents, frame_indices, timestamps_sec, fps, total_frames = encode_video_frames(
        model=model,
        video_path=config.video.input_path,
        image_size=config.video.image_size,
        color_mode=config.video.color_mode,
        batch_size=config.train.batch_size,
        device=device,
    )

    print("[4/8] latent 거리 기준 초기 프레임 집합 선택")
    selector = build_selector(config.selection)
    selection_result = validate_selection_result(selector.select(latents=latents, config=config.selection))

    print("[5/8] latent 거리 분산 local refinement")
    final_selected = selection_result.final_selected
    final_distances = selection_result.final_distances

    print("[6/8] 선택 프레임 이미지와 CSV 저장")
    selected_frame_indices = frame_indices[final_selected]
    selected_timestamps_sec = timestamps_sec[final_selected]
    selected_image_dir = output_dir / config.output.selected_frame_dir
    image_paths = export_selected_images(
        video_path=config.video.input_path,
        selected_frame_indices=selected_frame_indices,
        image_size=config.video.image_size,
        color_mode=config.video.color_mode,
        output_dir=selected_image_dir,
        save_original_size=config.output.save_selected_original_size,
        filename_prefix="selected",
    )

    uniform_frame_indices = compute_sample_indices(total_frames, config.selection.num_frames)
    uniform_image_dir = output_dir / config.output.uniform_frame_dir
    export_selected_images(
        video_path=config.video.input_path,
        selected_frame_indices=uniform_frame_indices,
        image_size=config.video.image_size,
        color_mode=config.video.color_mode,
        output_dir=uniform_image_dir,
        save_original_size=config.output.save_selected_original_size,
        filename_prefix="uniform",
    )
    frame_index_plot_path = output_dir / config.output.frame_index_plot
    save_frame_index_comparison_plot(
        output_path=frame_index_plot_path,
        selected_frame_indices=selected_frame_indices,
        uniform_frame_indices=uniform_frame_indices,
    )

    print("[7/8] 선택 프레임 CSV 저장")
    csv_path = output_dir / config.output.selected_csv
    write_selected_csv(
        output_path=csv_path,
        selected_candidate_orders=final_selected,
        selected_frame_indices=selected_frame_indices,
        selected_timestamps_sec=selected_timestamps_sec,
        cumulative=selection_result.cumulative,
        distances=final_distances,
        image_paths=image_paths,
    )

    print("[8/8] latent HTML, checkpoint, metrics 저장")
    checkpoint_path = output_dir / config.output.checkpoint
    latent_npz_path = output_dir / config.output.latent_npz
    metrics_path = output_dir / config.output.metrics_json
    latent_html_path = output_dir / config.output.latent_html

    save_checkpoint(checkpoint_path, model=model, history=history, config=config)
    save_latent_npz(
        output_path=latent_npz_path,
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        selected_candidate_orders=final_selected,
        distances=final_distances,
        fps=fps,
    )
    save_metrics_json(
        output_path=metrics_path,
        history=history,
        selection_result=selection_result,
        uniform_frame_indices=uniform_frame_indices,
        frame_indices=frame_indices,
        config=config,
    )
    visualizer = build_visualizer(config.visualization)
    visualizer.save(
        output_path=latent_html_path,
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        selected_candidate_orders=final_selected,
        config=config.visualization,
    )

    print(
        "완료: "
        f"initial_var={float(np.var(selection_result.initial_distances)):.6f}, "
        f"final_var={float(np.var(final_distances)):.6f}"
    )
    return {
        "selected_csv": csv_path,
        "selected_frame_dir": selected_image_dir,
        "uniform_frame_dir": uniform_image_dir,
        "frame_index_plot": frame_index_plot_path,
        "latent_html": latent_html_path,
        "checkpoint": checkpoint_path,
        "latent_npz": latent_npz_path,
        "metrics_json": metrics_path,
    }


def main() -> None:
    args = parse_args()
    config = parse_config(args.config)
    outputs = run_pipeline(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
