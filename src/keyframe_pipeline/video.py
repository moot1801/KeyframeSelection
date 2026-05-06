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

    processed_frames: list[np.ndarray] = []
    valid_indices: list[int] = []
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            continue
        _, model_frame = preprocess_frame(frame, image_size=image_size, color_mode=color_mode)
        processed_frames.append(model_frame)
        valid_indices.append(int(frame_index))

    capture.release()

    if len(processed_frames) != candidate_num_frames:
        raise RuntimeError(
            "일부 후보 프레임을 읽지 못했습니다. 다른 코덱이거나 seek 동작이 불안정할 수 있습니다."
        )

    frame_indices = np.asarray(valid_indices, dtype=np.int32)
    timestamps_sec = (
        frame_indices.astype(np.float32) / fps if fps > 0 else frame_indices.astype(np.float32)
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

    latents: list[np.ndarray] = []
    frame_indices: list[int] = []
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    frame_index = 0

    def flush_batch() -> None:
        if not batch_frames:
            return
        batch = torch.from_numpy(np.stack(batch_frames).astype(np.float32)).to(device)
        with torch.no_grad():
            latents.append(model.encode(batch).cpu().numpy())
        frame_indices.extend(batch_indices)
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
    return (
        np.concatenate(latents, axis=0).astype(np.float32),
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

    capture.release()
    return image_paths
