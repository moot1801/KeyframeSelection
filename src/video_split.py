from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상에서 균등 간격으로 프레임을 추출하고 오토인코더 학습용 데이터셋을 생성합니다."
    )
    parser.add_argument("--video-path", type=Path, required=True, help="입력 영상 경로")
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="추출할 총 프레임 수",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="오토인코더 입력 이미지 크기. 정사각형으로 리사이즈합니다.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("rgb", "grayscale"),
        default="rgb",
        help="저장할 프레임 색상 모드",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/frames"),
        help="프레임 데이터셋 저장 루트 디렉토리",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="출력 디렉토리 이름. 비우면 영상 이름 기반으로 자동 생성합니다.",
    )
    return parser.parse_args()


def build_dataset_name(video_path: Path, num_frames: int, image_size: int, color_mode: str) -> str:
    return f"{video_path.stem}_{num_frames}f_{image_size}px_{color_mode}"


def compute_sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("영상의 총 프레임 수를 확인할 수 없습니다.")
    if num_frames <= 0:
        raise ValueError("--num-frames 값은 1 이상이어야 합니다.")
    if num_frames > total_frames:
        raise ValueError(
            f"요청한 프레임 수({num_frames})가 전체 프레임 수({total_frames})보다 큽니다."
        )
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
    cv2.imwrite(str(output_path), writable)


def extract_frames(
    video_path: Path,
    num_frames: int,
    image_size: int,
    color_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_indices = compute_sample_indices(total_frames, num_frames)

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

    if len(processed_frames) != num_frames:
        raise RuntimeError(
            "일부 프레임을 읽지 못했습니다. 다른 코덱이거나 seek 동작이 불안정할 수 있습니다."
        )

    frame_indices = np.asarray(valid_indices, dtype=np.int32)
    timestamps_sec = (
        frame_indices.astype(np.float32) / fps if fps > 0 else frame_indices.astype(np.float32)
    )
    return np.stack(processed_frames), frame_indices, timestamps_sec, fps


def export_images(
    video_path: Path,
    frame_indices: np.ndarray,
    image_size: int,
    color_mode: str,
    images_dir: Path,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 다시 열 수 없습니다: {video_path}")

    for order, frame_index in enumerate(frame_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"프레임 저장 중 읽기 실패: index={frame_index}")
        preview_frame, _ = preprocess_frame(frame, image_size=image_size, color_mode=color_mode)
        save_preview_frame(
            preview_frame,
            images_dir / f"frame_{order:04d}_src_{int(frame_index):06d}.png",
            color_mode=color_mode,
        )

    capture.release()


def save_outputs(
    frames: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    fps: float,
    video_path: Path,
    color_mode: str,
    image_size: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    export_images(
        video_path=video_path,
        frame_indices=frame_indices,
        image_size=image_size,
        color_mode=color_mode,
        images_dir=images_dir,
    )

    dataset_path = output_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        frames=frames.astype(np.float32),
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        fps=np.array(fps, dtype=np.float32),
        source_video=np.array(str(video_path.resolve())),
        image_size=np.array(image_size, dtype=np.int32),
        color_mode=np.array(color_mode),
    )

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "source_video": str(video_path.resolve()),
        "num_frames": int(len(frames)),
        "frame_indices": frame_indices.tolist(),
        "timestamps_sec": timestamps_sec.tolist(),
        "fps": fps,
        "image_size": image_size,
        "color_mode": color_mode,
        "dataset_path": str(dataset_path.resolve()),
        "images_dir": str(images_dir.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset_path, metadata_path


def main() -> None:
    args = parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {args.video_path}")

    dataset_name = args.dataset_name or build_dataset_name(
        video_path=args.video_path,
        num_frames=args.num_frames,
        image_size=args.image_size,
        color_mode=args.color_mode,
    )
    output_dir = args.output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, frame_indices, timestamps_sec, fps = extract_frames(
        video_path=args.video_path,
        num_frames=args.num_frames,
        image_size=args.image_size,
        color_mode=args.color_mode,
    )
    dataset_path, metadata_path = save_outputs(
        frames=frames,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        fps=fps,
        video_path=args.video_path,
        color_mode=args.color_mode,
        image_size=args.image_size,
        output_dir=output_dir,
    )

    print(f"추출 프레임 수: {len(frames)}")
    print(f"데이터셋 저장 경로: {dataset_path}")
    print(f"메타데이터 저장 경로: {metadata_path}")


if __name__ == "__main__":
    main()
