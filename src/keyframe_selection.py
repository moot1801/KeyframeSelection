from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class VideoConfig:
    input_path: Path
    candidate_num_frames: int
    image_size: int
    color_mode: str


@dataclass(frozen=True)
class SelectionConfig:
    num_frames: int
    distance_metric: str
    include_endpoints: bool
    local_refine_iterations: int
    local_refine_window: int


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    num_workers: int
    device: str


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    selected_csv: str
    selected_frame_dir: str
    uniform_frame_dir: str
    frame_index_plot: str
    latent_html: str
    latent_npz: str
    checkpoint: str
    metrics_json: str
    save_selected_original_size: bool


@dataclass(frozen=True)
class VisualizationConfig:
    dimensions: int
    annotate_every: int
    show_all_candidates: bool
    show_selected_path: bool


@dataclass(frozen=True)
class PipelineConfig:
    video: VideoConfig
    selection: SelectionConfig
    model: ModelConfig
    train: TrainConfig
    output: OutputConfig
    visualization: VisualizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "영상 후보 프레임으로 autoencoder를 학습한 뒤 latent 거리 분산이 낮은 "
            "실제 영상 프레임 집합을 선택합니다."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML 설정 파일 경로")
    return parser.parse_args()


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value.rstrip()


def parse_yaml_scalar(value: str) -> Any:
    value = strip_inline_comment(value).strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none", "~"}:
        return None

    try:
        if "." not in value and "e" not in lower:
            return int(value)
        return float(value)
    except ValueError:
        return value


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise ValueError("YAML 최상위 값은 mapping이어야 합니다.")
        return loaded

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    with config_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
                raise ValueError(f"YAML 들여쓰기에는 탭을 사용할 수 없습니다: line {line_number}")

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if ":" not in line:
                raise ValueError(f"지원하지 않는 YAML 형식입니다: line {line_number}")

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"빈 YAML key입니다: line {line_number}")

            while indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]

            if value == "":
                child: dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = parse_yaml_scalar(value)

    return root


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"YAML에 '{name}' section이 필요합니다.")
    return section


def require_value(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ValueError(f"YAML '{section_name}.{key}' 값이 필요합니다.")
    return section[key]


def as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 값은 정수여야 합니다: {value}") from exc


def as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 값은 실수여야 합니다: {value}") from exc


def as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in {"true", "1", "yes", "y"}:
            return True
        if lower in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{name} 값은 boolean이어야 합니다: {value}")


def parse_config(config_path: Path) -> PipelineConfig:
    raw_config = load_yaml_config(config_path)

    video = require_section(raw_config, "video")
    selection = require_section(raw_config, "selection")
    model = require_section(raw_config, "model")
    train = require_section(raw_config, "train")
    output = require_section(raw_config, "output")
    visualization = require_section(raw_config, "visualization")

    config = PipelineConfig(
        video=VideoConfig(
            input_path=as_path(require_value(video, "video", "input_path")),
            candidate_num_frames=as_int(
                require_value(video, "video", "candidate_num_frames"),
                "video.candidate_num_frames",
            ),
            image_size=as_int(require_value(video, "video", "image_size"), "video.image_size"),
            color_mode=str(require_value(video, "video", "color_mode")),
        ),
        selection=SelectionConfig(
            num_frames=as_int(
                require_value(selection, "selection", "num_frames"),
                "selection.num_frames",
            ),
            distance_metric=str(require_value(selection, "selection", "distance_metric")),
            include_endpoints=as_bool(
                require_value(selection, "selection", "include_endpoints"),
                "selection.include_endpoints",
            ),
            local_refine_iterations=as_int(
                require_value(selection, "selection", "local_refine_iterations"),
                "selection.local_refine_iterations",
            ),
            local_refine_window=as_int(
                require_value(selection, "selection", "local_refine_window"),
                "selection.local_refine_window",
            ),
        ),
        model=ModelConfig(
            latent_dim=as_int(require_value(model, "model", "latent_dim"), "model.latent_dim")
        ),
        train=TrainConfig(
            epochs=as_int(require_value(train, "train", "epochs"), "train.epochs"),
            batch_size=as_int(require_value(train, "train", "batch_size"), "train.batch_size"),
            learning_rate=as_float(
                require_value(train, "train", "learning_rate"),
                "train.learning_rate",
            ),
            seed=as_int(require_value(train, "train", "seed"), "train.seed"),
            num_workers=as_int(
                require_value(train, "train", "num_workers"),
                "train.num_workers",
            ),
            device=str(require_value(train, "train", "device")),
        ),
        output=OutputConfig(
            output_dir=as_path(require_value(output, "output", "output_dir")),
            selected_csv=str(require_value(output, "output", "selected_csv")),
            selected_frame_dir=str(require_value(output, "output", "selected_frame_dir")),
            uniform_frame_dir=str(require_value(output, "output", "uniform_frame_dir")),
            frame_index_plot=str(require_value(output, "output", "frame_index_plot")),
            latent_html=str(require_value(output, "output", "latent_html")),
            latent_npz=str(require_value(output, "output", "latent_npz")),
            checkpoint=str(require_value(output, "output", "checkpoint")),
            metrics_json=str(require_value(output, "output", "metrics_json")),
            save_selected_original_size=as_bool(
                require_value(output, "output", "save_selected_original_size"),
                "output.save_selected_original_size",
            ),
        ),
        visualization=VisualizationConfig(
            dimensions=as_int(
                require_value(visualization, "visualization", "dimensions"),
                "visualization.dimensions",
            ),
            annotate_every=as_int(
                require_value(visualization, "visualization", "annotate_every"),
                "visualization.annotate_every",
            ),
            show_all_candidates=as_bool(
                require_value(visualization, "visualization", "show_all_candidates"),
                "visualization.show_all_candidates",
            ),
            show_selected_path=as_bool(
                require_value(visualization, "visualization", "show_selected_path"),
                "visualization.show_selected_path",
            ),
        ),
    )

    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    if not config.video.input_path.exists():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {config.video.input_path}")
    if config.video.color_mode not in {"rgb", "grayscale"}:
        raise ValueError("video.color_mode는 'rgb' 또는 'grayscale'이어야 합니다.")
    if config.video.image_size <= 0:
        raise ValueError("video.image_size는 1 이상이어야 합니다.")
    if config.video.candidate_num_frames <= 0:
        raise ValueError("video.candidate_num_frames는 1 이상이어야 합니다.")
    if config.selection.num_frames < 2:
        raise ValueError("selection.num_frames는 2 이상이어야 합니다.")
    if config.selection.distance_metric != "l2":
        raise ValueError("현재 selection.distance_metric은 'l2'만 지원합니다.")
    if config.selection.local_refine_iterations < 0:
        raise ValueError("selection.local_refine_iterations는 0 이상이어야 합니다.")
    if config.selection.local_refine_window < 0:
        raise ValueError("selection.local_refine_window는 0 이상이어야 합니다.")
    if config.model.latent_dim <= 0:
        raise ValueError("model.latent_dim은 1 이상이어야 합니다.")
    if config.train.epochs <= 0:
        raise ValueError("train.epochs는 1 이상이어야 합니다.")
    if config.train.batch_size <= 0:
        raise ValueError("train.batch_size는 1 이상이어야 합니다.")
    if config.train.learning_rate <= 0:
        raise ValueError("train.learning_rate는 0보다 커야 합니다.")
    if config.train.num_workers < 0:
        raise ValueError("train.num_workers는 0 이상이어야 합니다.")
    if config.train.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("train.device는 'auto', 'cpu', 'cuda' 중 하나여야 합니다.")
    if config.visualization.dimensions not in {2, 3}:
        raise ValueError("visualization.dimensions는 2 또는 3이어야 합니다.")
    if config.visualization.annotate_every < 0:
        raise ValueError("visualization.annotate_every는 0 이상이어야 합니다.")


def compute_downsampled_size(size: int, layers: int) -> int:
    value = size
    for _ in range(layers):
        value = ((value + 2 - 4) // 2) + 1
    if value <= 0:
        raise ValueError("입력 프레임 크기가 너무 작아서 4단계 다운샘플링을 처리할 수 없습니다.")
    return int(value)


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels: int, input_height: int, input_width: int, latent_dim: int) -> None:
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.feature_map_height = compute_downsampled_size(input_height, layers=4)
        self.feature_map_width = compute_downsampled_size(input_width, layers=4)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        flattened_size = 256 * self.feature_map_height * self.feature_map_width
        self.to_latent = nn.Linear(flattened_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        flattened = features.flatten(start_dim=1)
        return self.to_latent(flattened)

    def decode(self, latent: torch.Tensor, output_hw: tuple[int, int] | None = None) -> torch.Tensor:
        restored = self.from_latent(latent)
        restored = restored.view(-1, 256, self.feature_map_height, self.feature_map_width)
        reconstructed = self.decoder(restored)
        if output_hw is not None and reconstructed.shape[-2:] != output_hw:
            reconstructed = F.interpolate(
                reconstructed,
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )
        return reconstructed

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(inputs)
        reconstructed = self.decode(latent, output_hw=(inputs.shape[-2], inputs.shape[-1]))
        return reconstructed, latent


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device='cuda'로 설정했지만 CUDA를 사용할 수 없습니다.")
    return torch.device(device_name)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
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
    return np.stack(processed_frames), frame_indices, timestamps_sec, fps, total_frames


def train_autoencoder(
    model: ConvAutoEncoder,
    frames: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> list[float]:
    tensor_frames = torch.from_numpy(frames.astype(np.float32))
    dataloader = DataLoader(
        TensorDataset(tensor_frames),
        batch_size=min(config.batch_size, len(tensor_frames)),
        shuffle=True,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    history: list[float] = []

    model.to(device)
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        sample_count = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch.size(0)
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size

        average_loss = epoch_loss / max(sample_count, 1)
        history.append(float(average_loss))
        print(f"[Epoch {epoch + 1:03d}/{config.epochs:03d}] loss={average_loss:.6f}")

    return history


def encode_all_frames(
    model: ConvAutoEncoder,
    frames: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor_frames = torch.from_numpy(frames.astype(np.float32))
    dataloader = DataLoader(TensorDataset(tensor_frames), batch_size=batch_size, shuffle=False)

    model.eval()
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)
            latent = model.encode(batch).cpu().numpy()
            latents.append(latent)

    return np.concatenate(latents, axis=0).astype(np.float32)


def encode_video_frames(
    model: ConvAutoEncoder,
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


def adjacent_l2_distances(latents: np.ndarray) -> np.ndarray:
    if len(latents) < 2:
        return np.asarray([], dtype=np.float32)
    return np.linalg.norm(np.diff(latents, axis=0), axis=1).astype(np.float32)


def cumulative_path_distances(latents: np.ndarray) -> np.ndarray:
    distances = adjacent_l2_distances(latents)
    return np.concatenate([[0.0], np.cumsum(distances)]).astype(np.float32)


def choose_nearest_index(cumulative: np.ndarray, target: float, lower: int, upper: int) -> int:
    if lower > upper:
        raise ValueError("선택 가능한 후보 프레임 범위가 비어 있습니다.")
    nearest = int(np.searchsorted(cumulative, target, side="left"))
    nearest = min(max(nearest, lower), upper)
    scan_lower = max(lower, nearest - 4)
    scan_upper = min(upper, nearest + 4)
    candidates = range(scan_lower, scan_upper + 1)
    return min(candidates, key=lambda index: (abs(float(cumulative[index]) - target), index))


def initial_selection_by_arclength(
    cumulative: np.ndarray,
    num_frames: int,
    include_endpoints: bool,
) -> np.ndarray:
    candidate_count = len(cumulative)
    if num_frames > candidate_count:
        raise ValueError("최종 프레임 수가 선택 대상 전체 프레임 수보다 큽니다.")

    if include_endpoints:
        if num_frames == 2:
            return np.asarray([0, candidate_count - 1], dtype=np.int32)
        targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), num_frames)
        selected = [0]
        for order in range(1, num_frames - 1):
            lower = selected[-1] + 1
            upper = candidate_count - (num_frames - order)
            selected.append(choose_nearest_index(cumulative, float(targets[order]), lower, upper))
        selected.append(candidate_count - 1)
        return np.asarray(selected, dtype=np.int32)

    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), num_frames + 2)[1:-1]
    selected: list[int] = []
    for order, target in enumerate(targets):
        lower = selected[-1] + 1 if selected else 0
        upper = candidate_count - (num_frames - order)
        selected.append(choose_nearest_index(cumulative, float(target), lower, upper))
    return np.asarray(selected, dtype=np.int32)


def selected_distances(latents: np.ndarray, selected_candidate_orders: np.ndarray) -> np.ndarray:
    return adjacent_l2_distances(latents[selected_candidate_orders])


def distance_variance(latents: np.ndarray, selected_candidate_orders: np.ndarray) -> float:
    distances = selected_distances(latents, selected_candidate_orders)
    if len(distances) == 0:
        return 0.0
    return float(np.var(distances))


def refine_selection_by_local_search(
    latents: np.ndarray,
    selected_candidate_orders: np.ndarray,
    iterations: int,
    window: int,
    include_endpoints: bool,
) -> np.ndarray:
    if iterations == 0 or len(selected_candidate_orders) <= 2:
        return selected_candidate_orders.copy()

    selected = selected_candidate_orders.astype(np.int32).copy()
    fixed_positions = {0, len(selected) - 1} if include_endpoints else set()
    current_objective = distance_variance(latents, selected)

    for _ in range(iterations):
        changed = False
        for position in range(len(selected)):
            if position in fixed_positions:
                continue

            lower = int(selected[position - 1] + 1) if position > 0 else 0
            upper = (
                int(selected[position + 1] - 1)
                if position < len(selected) - 1
                else len(latents) - 1
            )
            if lower > upper:
                continue

            if window > 0:
                center = int(selected[position])
                lower = max(lower, center - window)
                upper = min(upper, center + window)

            best_index = int(selected[position])
            best_objective = current_objective
            for candidate_order in range(lower, upper + 1):
                if candidate_order == selected[position]:
                    continue
                trial = selected.copy()
                trial[position] = candidate_order
                objective = distance_variance(latents, trial)
                if objective + 1e-12 < best_objective:
                    best_objective = objective
                    best_index = candidate_order

            if best_index != selected[position]:
                selected[position] = best_index
                current_objective = best_objective
                changed = True

        if not changed:
            break

    return selected


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


def write_selected_csv(
    output_path: Path,
    selected_candidate_orders: np.ndarray,
    selected_frame_indices: np.ndarray,
    selected_timestamps_sec: np.ndarray,
    cumulative: np.ndarray,
    distances: np.ndarray,
    image_paths: list[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_distances = np.concatenate([[np.nan], distances]).astype(np.float32)
    next_distances = np.concatenate([distances, [np.nan]]).astype(np.float32)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "selection_order",
                "candidate_order",
                "frame_index",
                "timestamp_sec",
                "cumulative_latent_distance",
                "prev_latent_distance",
                "next_latent_distance",
                "image_path",
            ],
        )
        writer.writeheader()
        for order, candidate_order in enumerate(selected_candidate_orders):
            writer.writerow(
                {
                    "selection_order": order,
                    "candidate_order": int(candidate_order),
                    "frame_index": int(selected_frame_indices[order]),
                    "timestamp_sec": f"{float(selected_timestamps_sec[order]):.6f}",
                    "cumulative_latent_distance": f"{float(cumulative[candidate_order]):.6f}",
                    "prev_latent_distance": (
                        ""
                        if np.isnan(previous_distances[order])
                        else f"{float(previous_distances[order]):.6f}"
                    ),
                    "next_latent_distance": (
                        ""
                        if np.isnan(next_distances[order])
                        else f"{float(next_distances[order]):.6f}"
                    ),
                    "image_path": str(image_paths[order]),
                }
            )


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


def save_checkpoint(
    output_path: Path,
    model: ConvAutoEncoder,
    history: list[float],
    config: PipelineConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": model.in_channels,
                "input_height": model.input_height,
                "input_width": model.input_width,
                "latent_dim": model.latent_dim,
            },
            "train_loss": history,
            "source_video": str(config.video.input_path),
        },
        output_path,
    )


def save_latent_npz(
    output_path: Path,
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    selected_candidate_orders: np.ndarray,
    distances: np.ndarray,
    fps: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        latents=latents.astype(np.float32),
        frame_indices=frame_indices.astype(np.int32),
        timestamps_sec=timestamps_sec.astype(np.float32),
        selected_candidate_orders=selected_candidate_orders.astype(np.int32),
        selected_frame_indices=frame_indices[selected_candidate_orders].astype(np.int32),
        selected_distances=distances.astype(np.float32),
        fps=np.array(fps, dtype=np.float32),
    )


def save_metrics_json(
    output_path: Path,
    history: list[float],
    initial_selected: np.ndarray,
    final_selected: np.ndarray,
    initial_distances: np.ndarray,
    final_distances: np.ndarray,
    uniform_frame_indices: np.ndarray,
    frame_indices: np.ndarray,
    config: PipelineConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_video": str(config.video.input_path),
        "training_candidate_num_frames": config.video.candidate_num_frames,
        "selection_source_frame_count": int(len(frame_indices)),
        "selected_num_frames": config.selection.num_frames,
        "initial_selected_frame_indices": frame_indices[initial_selected].astype(int).tolist(),
        "final_selected_frame_indices": frame_indices[final_selected].astype(int).tolist(),
        "uniform_frame_indices": uniform_frame_indices.astype(int).tolist(),
        "initial_distance_mean": float(np.mean(initial_distances)) if len(initial_distances) else 0.0,
        "initial_distance_variance": float(np.var(initial_distances)) if len(initial_distances) else 0.0,
        "final_distance_mean": float(np.mean(final_distances)) if len(final_distances) else 0.0,
        "final_distance_variance": float(np.var(final_distances)) if len(final_distances) else 0.0,
        "final_distance_std": float(np.std(final_distances)) if len(final_distances) else 0.0,
        "final_distances": final_distances.astype(float).tolist(),
        "epochs": config.train.epochs,
        "final_train_loss": history[-1] if history else None,
        "latent_dim": config.model.latent_dim,
        "selection_method": "all_video_frame_latent_arclength_with_local_variance_refinement",
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def save_latent_html(
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
                    marker={"size": 6, "color": "#f97316", "line": {"color": "#7f1d1d", "width": 2}},
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
                    marker={"size": 8, "color": "#f97316", "line": {"color": "#7f1d1d", "width": 1}},
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


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    set_seed(config.train.seed)
    ensure_plotly_available()

    output_dir = config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] 학습용 후보 프레임 추출")
    training_frames, _, _, _, _ = extract_candidate_frames(
        video_path=config.video.input_path,
        candidate_num_frames=config.video.candidate_num_frames,
        image_size=config.video.image_size,
        color_mode=config.video.color_mode,
    )

    print("[2/8] autoencoder 학습")
    device = resolve_device(config.train.device)
    model = ConvAutoEncoder(
        in_channels=training_frames.shape[1],
        input_height=training_frames.shape[-2],
        input_width=training_frames.shape[-1],
        latent_dim=config.model.latent_dim,
    )
    history = train_autoencoder(model=model, frames=training_frames, config=config.train, device=device)

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
    cumulative = cumulative_path_distances(latents)
    initial_selected = initial_selection_by_arclength(
        cumulative=cumulative,
        num_frames=config.selection.num_frames,
        include_endpoints=config.selection.include_endpoints,
    )
    initial_distances = selected_distances(latents, initial_selected)

    print("[5/8] latent 거리 분산 local refinement")
    final_selected = refine_selection_by_local_search(
        latents=latents,
        selected_candidate_orders=initial_selected,
        iterations=config.selection.local_refine_iterations,
        window=config.selection.local_refine_window,
        include_endpoints=config.selection.include_endpoints,
    )
    final_distances = selected_distances(latents, final_selected)

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
        cumulative=cumulative,
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
        initial_selected=initial_selected,
        final_selected=final_selected,
        initial_distances=initial_distances,
        final_distances=final_distances,
        uniform_frame_indices=uniform_frame_indices,
        frame_indices=frame_indices,
        config=config,
    )
    save_latent_html(
        output_path=latent_html_path,
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        selected_candidate_orders=final_selected,
        config=config.visualization,
    )

    print(
        "완료: "
        f"initial_var={float(np.var(initial_distances)):.6f}, "
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


if __name__ == "__main__":
    main()
