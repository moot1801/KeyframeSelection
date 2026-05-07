from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class VideoFrameBatch:
    frames: np.ndarray
    frame_indices: np.ndarray
    timestamps_sec: np.ndarray
    fps: float
    total_frames: int


def should_log_progress(current: int, total: int) -> bool:
    if current <= 0:
        return False
    if total <= 0:
        return current == 1 or current % 100 == 0
    interval = max(1, total // 10)
    return current == 1 or current == total or current % interval == 0


def compute_sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("영상의 총 프레임 수를 확인할 수 없습니다.")
    if num_frames <= 0:
        raise ValueError("추출할 프레임 수는 1 이상이어야 합니다.")
    if num_frames > total_frames:
        raise ValueError(f"요청한 후보 프레임 수({num_frames})가 전체 프레임 수({total_frames})보다 큽니다.")
    return np.linspace(0, total_frames - 1, num=num_frames, dtype=int)


def preprocess_frame(frame_bgr: np.ndarray, image_size: int, color_mode: str) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if color_mode == "rgb":
        display_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        model_frame = np.transpose(display_frame.astype(np.float32) / 255.0, (2, 0, 1))
        return display_frame, model_frame

    gray_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    model_frame = gray_frame.astype(np.float32)[None, ...] / 255.0
    return gray_frame, model_frame


def save_preview_frame(frame: np.ndarray, output_path: Path, color_mode: str) -> None:
    if color_mode == "rgb":
        writable = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        writable = frame
    if not cv2.imwrite(str(output_path), writable):
        raise RuntimeError(f"이미지 저장에 실패했습니다: {output_path}")


def extract_candidate_frames(
    video_path: Path,
    candidate_num_frames: int,
    image_size: int,
    color_mode: str,
) -> VideoFrameBatch:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_indices = compute_sample_indices(total_frames, candidate_num_frames)
    print(
        "  - video opened: "
        f"path={video_path}, total_frames={total_frames}, fps={fps:.3f}, "
        f"candidate_num_frames={candidate_num_frames}, image_size={image_size}, color_mode={color_mode}"
    )

    processed_frames: list[np.ndarray] = []
    valid_indices: list[int] = []
    for sample_order, frame_index in enumerate(sample_indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            print(f"  - candidate read failed: order={sample_order}, frame_index={int(frame_index)}")
            continue
        _, model_frame = preprocess_frame(frame, image_size=image_size, color_mode=color_mode)
        processed_frames.append(model_frame)
        valid_indices.append(int(frame_index))
        if should_log_progress(sample_order, candidate_num_frames):
            print(
                "  - candidate extraction progress: "
                f"{sample_order}/{candidate_num_frames}, frame_index={int(frame_index)}"
            )

    capture.release()

    if len(processed_frames) != candidate_num_frames:
        raise RuntimeError(
            "일부 후보 프레임을 읽지 못했습니다. 다른 코덱이거나 seek 동작이 불안정할 수 있습니다."
        )

    frame_indices = np.asarray(valid_indices, dtype=np.int32)
    timestamps_sec = (
        frame_indices.astype(np.float32) / fps if fps > 0 else frame_indices.astype(np.float32)
    )
    print(
        "  - candidate extraction complete: "
        f"frames_shape={np.stack(processed_frames).shape}, "
        f"frame_index_range={int(frame_indices[0])}-{int(frame_indices[-1])}"
    )
    return VideoFrameBatch(
        frames=np.stack(processed_frames),
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        fps=fps,
        total_frames=total_frames,
    )


def encode_video_frames(
    model: nn.Module,
    video_path: Path,
    image_size: int,
    color_mode: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    model.eval()
    print(
        "  - encoding setup: "
        f"path={video_path}, total_frames={total_frames}, fps={fps:.3f}, "
        f"batch_size={batch_size}, image_size={image_size}, color_mode={color_mode}, device={device}"
    )

    latents: list[np.ndarray] = []
    frame_indices: list[int] = []
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    frame_index = 0
    encoded_batches = 0
    encoded_frame_count = 0

    def flush_batch() -> None:
        nonlocal encoded_batches, encoded_frame_count
        if not batch_frames:
            return
        current_batch_size = len(batch_frames)
        batch = torch.from_numpy(np.stack(batch_frames).astype(np.float32)).to(device)
        with torch.no_grad():
            latents.append(model.encode(batch).cpu().numpy())
        frame_indices.extend(batch_indices)
        encoded_batches += 1
        encoded_frame_count += current_batch_size
        if should_log_progress(encoded_frame_count, total_frames):
            print(
                "  - latent encoding progress: "
                f"encoded_frames={encoded_frame_count}/{total_frames}, batches={encoded_batches}"
            )
        batch_frames.clear()
        batch_indices.clear()

    while True:
        success, frame = capture.read()
        if not success:
            break
        _, model_frame = preprocess_frame(frame, image_size=image_size, color_mode=color_mode)
        batch_frames.append(model_frame)
        batch_indices.append(frame_index)
        if len(batch_frames) >= batch_size:
            flush_batch()
        frame_index += 1

    flush_batch()
    capture.release()

    if not latents:
        raise RuntimeError(f"선택용 전체 프레임을 읽지 못했습니다: {video_path}")

    all_frame_indices = np.asarray(frame_indices, dtype=np.int32)
    timestamps_sec = (
        all_frame_indices.astype(np.float32) / fps
        if fps > 0
        else all_frame_indices.astype(np.float32)
    )
    observed_total_frames = len(all_frame_indices)
    if total_frames <= 0:
        total_frames = observed_total_frames
    latent_array = np.concatenate(latents, axis=0).astype(np.float32)
    print(
        "  - latent encoding complete: "
        f"encoded_frames={observed_total_frames}, latent_shape={latent_array.shape}, fps={fps:.3f}"
    )
    return (
        latent_array,
        all_frame_indices,
        timestamps_sec,
        fps,
        total_frames,
    )


def export_selected_images(
    video_path: Path,
    selected_frame_indices: np.ndarray,
    image_size: int,
    color_mode: str,
    output_dir: Path,
    save_original_size: bool,
    filename_prefix: str = "selected",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 다시 열 수 없습니다: {video_path}")

    image_paths: list[Path] = []
    total_selected = len(selected_frame_indices)
    print(
        "  - image export setup: "
        f"count={total_selected}, output_dir={output_dir}, save_original_size={save_original_size}, "
        f"prefix={filename_prefix}"
    )
    for order, frame_index in enumerate(selected_frame_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"선택 프레임 저장 중 읽기 실패: index={frame_index}")

        output_path = output_dir / f"{filename_prefix}_{order:04d}_src_{int(frame_index):06d}.png"
        if save_original_size:
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"이미지 저장에 실패했습니다: {output_path}")
        else:
            preview_frame, _ = preprocess_frame(frame, image_size=image_size, color_mode=color_mode)
            save_preview_frame(preview_frame, output_path, color_mode=color_mode)
        image_paths.append(output_path)
        if should_log_progress(order + 1, total_selected):
            print(
                "  - image export progress: "
                f"{order + 1}/{total_selected}, frame_index={int(frame_index)}, path={output_path}"
            )

    capture.release()
    print(f"  - image export complete: output_dir={output_dir}")
    return image_paths


def even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def map_frame_to_x(frame_index: int, total_frames: int, left: int, right: int) -> int:
    if total_frames <= 1:
        return left
    ratio = min(max(frame_index / float(total_frames - 1), 0.0), 1.0)
    return int(round(left + ratio * (right - left)))


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (30, 41, 59),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def build_timeline_panel(
    width: int,
    height: int,
    current_frame_index: int,
    total_frames: int,
    fps: float,
    selected_frame_indices: np.ndarray,
    uniform_frame_indices: np.ndarray,
) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_left = 76
    plot_right = width - 34
    selected_y = int(height * 0.38)
    uniform_y = int(height * 0.56)
    axis_y = int(height * 0.75)
    current_x = map_frame_to_x(current_frame_index, total_frames, plot_left, plot_right)

    draw_text(panel, "Timeline comparison", (24, 42), scale=0.72, thickness=2)
    current_time = current_frame_index / fps if fps > 0 else float(current_frame_index)
    total_time = (total_frames - 1) / fps if fps > 0 and total_frames > 0 else float(max(total_frames - 1, 0))
    draw_text(
        panel,
        f"frame {current_frame_index:,} / {max(total_frames - 1, 0):,}   time {current_time:.2f}s / {total_time:.2f}s",
        (24, 72),
        scale=0.48,
    )

    cv2.line(panel, (plot_left, selected_y), (plot_right, selected_y), (226, 232, 240), 2)
    cv2.line(panel, (plot_left, uniform_y), (plot_right, uniform_y), (226, 232, 240), 2)
    cv2.line(panel, (plot_left, axis_y), (plot_right, axis_y), (148, 163, 184), 1)
    draw_text(panel, "selected", (16, selected_y + 5), scale=0.45, color=(185, 28, 28))
    draw_text(panel, "uniform", (16, uniform_y + 5), scale=0.45, color=(37, 99, 235))

    tick_count = 5
    for tick in range(tick_count):
        ratio = tick / float(tick_count - 1)
        frame_index = int(round(ratio * max(total_frames - 1, 0)))
        x = map_frame_to_x(frame_index, total_frames, plot_left, plot_right)
        cv2.line(panel, (x, axis_y - 6), (x, axis_y + 6), (100, 116, 139), 1)
        draw_text(panel, f"{frame_index:,}", (max(6, x - 22), axis_y + 28), scale=0.36, color=(71, 85, 105))

    for frame_index in uniform_frame_indices.astype(int):
        x = map_frame_to_x(int(frame_index), total_frames, plot_left, plot_right)
        cv2.circle(panel, (x, uniform_y), 5, (37, 99, 235), -1)
        cv2.circle(panel, (x, uniform_y), 5, (30, 64, 175), 1)

    for order, frame_index in enumerate(selected_frame_indices.astype(int)):
        x = map_frame_to_x(int(frame_index), total_frames, plot_left, plot_right)
        cv2.circle(panel, (x, selected_y), 6, (220, 38, 38), -1)
        cv2.circle(panel, (x, selected_y), 6, (127, 29, 29), 1)
        if order == 0 or order == len(selected_frame_indices) - 1 or order % 5 == 0:
            draw_text(panel, str(order), (x - 6, selected_y - 12), scale=0.34, color=(127, 29, 29))

    cv2.line(panel, (current_x, 92), (current_x, height - 42), (15, 23, 42), 2)
    cv2.rectangle(panel, (current_x - 44, 84), (current_x + 44, 108), (15, 23, 42), -1)
    draw_text(panel, "current", (current_x - 33, 101), scale=0.4, color=(255, 255, 255))

    legend_y = height - 68
    cv2.circle(panel, (26, legend_y), 5, (220, 38, 38), -1)
    draw_text(panel, "selected keyframes", (42, legend_y + 5), scale=0.42)
    cv2.circle(panel, (26, legend_y + 24), 5, (37, 99, 235), -1)
    draw_text(panel, "uniform samples", (42, legend_y + 29), scale=0.42)
    return panel


def save_timeline_comparison_video(
    video_path: Path,
    selected_frame_indices: np.ndarray,
    uniform_frame_indices: np.ndarray,
    output_path: Path,
    output_height: int = 720,
    timeline_width: int = 760,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"타임라인 비교 영상을 만들기 위해 영상을 열 수 없습니다: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError(f"영상 크기를 확인할 수 없습니다: {video_path}")

    writer_fps = fps if fps > 0 else 30.0
    panel_height = even(output_height)
    video_width = even(int(round(panel_height * source_width / float(source_height))))
    panel_width = even(timeline_width)
    output_size = (video_width + panel_width, panel_height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        writer_fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"타임라인 비교 영상 writer를 열 수 없습니다: {output_path}")

    print(
        "  - timeline video setup: "
        f"source={video_path}, output={output_path}, fps={writer_fps:.3f}, "
        f"total_frames={total_frames}, output_size={output_size[0]}x{output_size[1]}"
    )

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (video_width, panel_height), interpolation=cv2.INTER_AREA)
        timeline_panel = build_timeline_panel(
            width=panel_width,
            height=panel_height,
            current_frame_index=frame_index,
            total_frames=total_frames,
            fps=fps,
            selected_frame_indices=selected_frame_indices,
            uniform_frame_indices=uniform_frame_indices,
        )
        writer.write(np.concatenate([resized_frame, timeline_panel], axis=1))
        frame_index += 1
        if should_log_progress(frame_index, total_frames):
            print(f"  - timeline video progress: {frame_index}/{total_frames}")

    capture.release()
    writer.release()
    print(f"  - timeline video complete: frames={frame_index}, path={output_path}")


def read_frame_at(capture: cv2.VideoCapture, frame_index: int, video_path: Path) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"비교 영상 생성 중 프레임 읽기 실패: path={video_path}, frame_index={frame_index}")
    return frame


def prepare_comparison_frame(
    frame: np.ndarray,
    size: tuple[int, int],
    title: str,
    frame_index: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    overlay = resized.copy()
    cv2.rectangle(overlay, (0, 0), (size[0], 44), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.82, resized, 0.18, 0.0, resized)
    cv2.rectangle(resized, (12, 10), (28, 26), color, -1)
    draw_text(
        resized,
        f"{title}   frame={int(frame_index):,}",
        (38, 27),
        scale=0.52,
        color=(255, 255, 255),
        thickness=1,
    )
    return resized


def save_selected_vs_uniform_video(
    video_path: Path,
    selected_frame_indices: np.ndarray,
    uniform_frame_indices: np.ndarray,
    output_path: Path,
    interval_sec: float = 0.2,
    output_height: int = 720,
) -> None:
    if interval_sec <= 0:
        raise ValueError("selected/uniform 비교 영상 interval_sec는 0보다 커야 합니다.")
    if len(selected_frame_indices) != len(uniform_frame_indices):
        raise ValueError(
            "selected/uniform 비교 영상은 두 프레임 집합의 길이가 같아야 합니다: "
            f"selected={len(selected_frame_indices)}, uniform={len(uniform_frame_indices)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"selected/uniform 비교 영상을 만들기 위해 영상을 열 수 없습니다: {video_path}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError(f"영상 크기를 확인할 수 없습니다: {video_path}")

    pane_height = even(output_height)
    pane_width = even(int(round(pane_height * source_width / float(source_height))))
    output_size = (pane_width * 2, pane_height)
    writer_fps = 1.0 / interval_sec
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        writer_fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"selected/uniform 비교 영상 writer를 열 수 없습니다: {output_path}")

    pair_count = len(selected_frame_indices)
    print(
        "  - selected/uniform video setup: "
        f"source={video_path}, output={output_path}, pairs={pair_count}, "
        f"interval_sec={interval_sec:.3f}, fps={writer_fps:.3f}, "
        f"output_size={output_size[0]}x{output_size[1]}"
    )

    pane_size = (pane_width, pane_height)
    for order, (selected_index, uniform_index) in enumerate(
        zip(selected_frame_indices.astype(int), uniform_frame_indices.astype(int)),
        start=1,
    ):
        selected_frame = read_frame_at(capture, int(selected_index), video_path)
        uniform_frame = read_frame_at(capture, int(uniform_index), video_path)
        selected_panel = prepare_comparison_frame(
            selected_frame,
            size=pane_size,
            title=f"KS selected #{order - 1}",
            frame_index=int(selected_index),
            color=(220, 38, 38),
        )
        uniform_panel = prepare_comparison_frame(
            uniform_frame,
            size=pane_size,
            title=f"Uniform #{order - 1}",
            frame_index=int(uniform_index),
            color=(37, 99, 235),
        )
        writer.write(np.concatenate([selected_panel, uniform_panel], axis=1))
        if should_log_progress(order, pair_count):
            print(
                "  - selected/uniform video progress: "
                f"{order}/{pair_count}, selected_frame={int(selected_index)}, uniform_frame={int(uniform_index)}"
            )

    capture.release()
    writer.release()
    print(f"  - selected/uniform video complete: frames={pair_count}, path={output_path}")
