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


def create_unique_output_dir(output_dir: Path) -> Path:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    parent = output_dir.parent
    name = output_dir.name
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = parent / f"{name}_{index:03d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue

    raise RuntimeError(f"사용 가능한 출력 디렉터리를 찾지 못했습니다: {output_dir}")


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
        save_selected_vs_uniform_video,
        save_timeline_comparison_video,
    )
    from keyframe_pipeline.visualizers import build_visualizer, save_frame_index_comparison_plot

    set_seed(config.train.seed)

    output_dir = create_unique_output_dir(config.output.output_dir)
    if output_dir != config.output.output_dir:
        print(f"출력 디렉터리가 이미 있어 새 디렉터리를 사용합니다: {output_dir}")
    print(f"출력 디렉터리: {output_dir}")
    print(
        "설정 요약: "
        f"video={config.video.input_path}, model={config.model.module}.{config.model.class_name}, "
        f"selection={config.selection.module}.{config.selection.class_name}, "
        f"visualization={config.visualization.module}.{config.visualization.class_name}"
    )

    print("[1/8] 학습용 후보 프레임 추출")
    training_batch = extract_candidate_frames(
        video_path=config.video.input_path,
        candidate_num_frames=config.video.candidate_num_frames,
        image_size=config.video.image_size,
        color_mode=config.video.color_mode,
    )
    print(
        "  - training batch ready: "
        f"frames_shape={training_batch.frames.shape}, "
        f"frame_index_range={int(training_batch.frame_indices[0])}-{int(training_batch.frame_indices[-1])}"
    )

    print("[2/8] autoencoder 학습")
    device = resolve_device(config.train.device)
    model = build_model(
        config=config.model,
        in_channels=training_batch.frames.shape[1],
        input_height=training_batch.frames.shape[-2],
        input_width=training_batch.frames.shape[-1],
    )
    print(
        "  - model built: "
        f"class={model.__class__.__name__}, in_channels={training_batch.frames.shape[1]}, "
        f"input_size={training_batch.frames.shape[-2]}x{training_batch.frames.shape[-1]}, "
        f"latent_dim={config.model.latent_dim}, device={device}"
    )
    history = train_autoencoder(
        model=model,
        frames=training_batch.frames,
        train_config=config.train,
        optimizer_config=config.optimizer,
        loss_config=config.loss,
        device=device,
    )
    print(f"  - training complete: final_loss={history[-1] if history else None}")

    print("[3/8] 선택용 전체 영상 프레임 latent 추출")
    latents, frame_indices, timestamps_sec, fps, total_frames = encode_video_frames(
        model=model,
        video_path=config.video.input_path,
        image_size=config.video.image_size,
        color_mode=config.video.color_mode,
        batch_size=config.train.batch_size,
        device=device,
    )
    print(
        "  - selection source ready: "
        f"latents_shape={latents.shape}, frame_count={len(frame_indices)}, "
        f"frame_index_range={int(frame_indices[0])}-{int(frame_indices[-1])}, fps={fps:.3f}"
    )

    print("[4/8] latent 거리 기준 초기 프레임 집합 선택")
    selector = build_selector(config.selection)
    print(f"  - selector built: class={selector.__class__.__name__}")
    selection_result = validate_selection_result(selector.select(latents=latents, config=config.selection))

    print("[5/8] latent 거리 분산 local refinement")
    final_selected = selection_result.final_selected
    final_distances = selection_result.final_distances
    print(
        "  - refinement result: "
        f"initial_var={float(np.var(selection_result.initial_distances)):.6f}, "
        f"final_var={float(np.var(final_distances)):.6f}, "
        f"selected_count={len(final_selected)}"
    )

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
    print(f"  - selected images saved: count={len(image_paths)}, dir={selected_image_dir}")

    uniform_frame_indices = compute_sample_indices(total_frames, len(final_selected))
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
    print(f"  - uniform comparison images saved: count={len(uniform_frame_indices)}, dir={uniform_image_dir}")
    selected_vs_uniform_video_path = output_dir / config.output.selected_vs_uniform_video
    save_selected_vs_uniform_video(
        video_path=config.video.input_path,
        selected_frame_indices=selected_frame_indices,
        uniform_frame_indices=uniform_frame_indices,
        output_path=selected_vs_uniform_video_path,
        interval_sec=config.output.selected_vs_uniform_interval_sec,
    )
    print(f"  - selected/uniform comparison video saved: {selected_vs_uniform_video_path}")
    frame_index_plot_path = output_dir / config.output.frame_index_plot
    save_frame_index_comparison_plot(
        output_path=frame_index_plot_path,
        selected_frame_indices=selected_frame_indices,
        uniform_frame_indices=uniform_frame_indices,
    )
    print(f"  - frame index comparison plot saved: {frame_index_plot_path}")
    timeline_video_path = output_dir / config.output.timeline_comparison_video
    save_timeline_comparison_video(
        video_path=config.video.input_path,
        selected_frame_indices=selected_frame_indices,
        uniform_frame_indices=uniform_frame_indices,
        output_path=timeline_video_path,
    )
    print(f"  - timeline comparison video saved: {timeline_video_path}")

    print("[7/8] 선택 프레임 CSV 저장")
    csv_path = output_dir / config.output.selected_csv
    write_selected_csv(
        output_path=csv_path,
        selected_candidate_orders=final_selected,
        selected_frame_indices=selected_frame_indices,
        selected_timestamps_sec=selected_timestamps_sec,
        cumulative=selection_result.cumulative,
        distances=final_distances,
        latents=latents,
        image_paths=image_paths,
        precluster_summary=selection_result.precluster_summary,
    )
    print(f"  - selected frame CSV saved: {csv_path}")

    print("[8/8] latent HTML, checkpoint, metrics 저장")
    checkpoint_path = output_dir / config.output.checkpoint
    latent_npz_path = output_dir / config.output.latent_npz
    metrics_path = output_dir / config.output.metrics_json
    latent_html_path = output_dir / config.output.latent_html

    save_checkpoint(checkpoint_path, model=model, history=history, config=config)
    print(f"  - checkpoint saved: {checkpoint_path}")
    save_latent_npz(
        output_path=latent_npz_path,
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        selected_candidate_orders=final_selected,
        distances=final_distances,
        fps=fps,
        precluster_summary=selection_result.precluster_summary,
    )
    print(f"  - latent NPZ saved: {latent_npz_path}")
    save_metrics_json(
        output_path=metrics_path,
        history=history,
        selection_result=selection_result,
        uniform_frame_indices=uniform_frame_indices,
        frame_indices=frame_indices,
        latents=latents,
        config=config,
    )
    print(f"  - metrics JSON saved: {metrics_path}")
    visualizer = build_visualizer(config.visualization)
    print(f"  - visualizer built: class={visualizer.__class__.__name__}")
    visualizer.save(
        output_path=latent_html_path,
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        selected_candidate_orders=final_selected,
        config=config.visualization,
    )
    print(f"  - latent HTML saved: {latent_html_path}")

    print(
        "완료: "
        f"initial_var={float(np.var(selection_result.initial_distances)):.6f}, "
        f"final_var={float(np.var(final_distances)):.6f}"
    )
    return {
        "selected_csv": csv_path,
        "selected_frame_dir": selected_image_dir,
        "uniform_frame_dir": uniform_image_dir,
        "selected_vs_uniform_video": selected_vs_uniform_video_path,
        "frame_index_plot": frame_index_plot_path,
        "timeline_comparison_video": timeline_video_path,
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
