from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoConfig:
    input_path: Path
    candidate_num_frames: int
    image_size: int
    color_mode: str


@dataclass(frozen=True)
class SelectionConfig:
    name: str
    module: str
    class_name: str
    kwargs: dict[str, Any]
    num_frames: int
    distance_metric: str
    include_endpoints: bool
    local_refine_iterations: int
    local_refine_window: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    module: str
    class_name: str
    latent_dim: int
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    module: str
    class_name: str
    learning_rate: float
    weight_decay: float
    momentum: float
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class LossConfig:
    name: str
    module: str
    class_name: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
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
    name: str
    module: str
    class_name: str
    kwargs: dict[str, Any]
    dimensions: int
    annotate_every: int
    show_all_candidates: bool
    show_selected_path: bool


@dataclass(frozen=True)
class PipelineConfig:
    video: VideoConfig
    selection: SelectionConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    train: TrainConfig
    output: OutputConfig
    visualization: VisualizationConfig


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


def optional_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"YAML '{name}' section은 mapping이어야 합니다.")
    return section


def require_value(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ValueError(f"YAML '{section_name}.{key}' 값이 필요합니다.")
    return section[key]


def optional_value(section: dict[str, Any], key: str, default: Any) -> Any:
    return section[key] if key in section else default


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


def as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        items = [item.strip() for item in stripped.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"{name} 값은 정수 목록이어야 합니다: {value}")

    parsed = tuple(as_int(item, name) for item in items)
    if not parsed:
        raise ValueError(f"{name} 값은 비어 있을 수 없습니다.")
    return parsed


def as_kwargs(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} 값은 mapping이어야 합니다: {value}")
    return dict(value)


def normalize_model_kwargs(model_section: dict[str, Any]) -> dict[str, Any]:
    kwargs = as_kwargs(optional_value(model_section, "kwargs", {}), "model.kwargs")
    legacy_keys = ("encoder_channels", "activation", "output_activation")
    for key in legacy_keys:
        if key in model_section and key not in kwargs:
            kwargs[key] = model_section[key]
    module = str(optional_value(model_section, "module", "keyframe_pipeline.models.conv_autoencoder"))
    class_name = str(optional_value(model_section, "class_name", "ConvAutoEncoder"))
    if module == "keyframe_pipeline.models.conv_autoencoder" and class_name == "ConvAutoEncoder":
        kwargs.setdefault("encoder_channels", "32,64,128,256")
        kwargs.setdefault("activation", "relu")
        kwargs.setdefault("output_activation", "sigmoid")
    if "encoder_channels" in kwargs:
        kwargs["encoder_channels"] = as_int_tuple(kwargs["encoder_channels"], "model.kwargs.encoder_channels")
    return kwargs


def default_optimizer_class_name(name: str) -> str:
    mapping = {
        "adam": "AdamOptimizerStrategy",
        "adamw": "AdamWOptimizerStrategy",
        "sgd": "SGDOptimizerStrategy",
    }
    return mapping.get(name, "AdamOptimizerStrategy")


def default_loss_class_name(name: str) -> str:
    mapping = {
        "mse": "MSELossStrategy",
        "l1": "L1LossStrategy",
        "smooth_l1": "SmoothL1LossStrategy",
    }
    return mapping.get(name, "MSELossStrategy")


def default_visualization_class_name(name: str) -> str:
    mapping = {
        "plotly_latent": "PlotlyLatentVisualizationStrategy",
        "none": "DisabledVisualizationStrategy",
    }
    return mapping.get(name, "PlotlyLatentVisualizationStrategy")


def parse_config(config_path: Path) -> PipelineConfig:
    raw_config = load_yaml_config(config_path)

    video = require_section(raw_config, "video")
    selection = require_section(raw_config, "selection")
    model = require_section(raw_config, "model")
    train = require_section(raw_config, "train")
    output = require_section(raw_config, "output")
    visualization = require_section(raw_config, "visualization")
    optimizer = optional_section(raw_config, "optimizer")
    loss = optional_section(raw_config, "loss")

    legacy_learning_rate = optional_value(train, "learning_rate", 0.001)
    optimizer_name = str(optional_value(optimizer, "name", "adam"))
    loss_name = str(optional_value(loss, "name", "mse"))
    visualization_name = str(optional_value(visualization, "name", "plotly_latent"))
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
            name=str(optional_value(selection, "name", "arclength_local_refine")),
            module=str(
                optional_value(
                    selection,
                    "module",
                    "keyframe_pipeline.selectors.arclength_local_refine",
                )
            ),
            class_name=str(
                optional_value(
                    selection,
                    "class_name",
                    "ArclengthLocalRefineSelectionStrategy",
                )
            ),
            kwargs=as_kwargs(optional_value(selection, "kwargs", {}), "selection.kwargs"),
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
            name=str(optional_value(model, "name", "conv_autoencoder")),
            module=str(
                optional_value(
                    model,
                    "module",
                    "keyframe_pipeline.models.conv_autoencoder",
                )
            ),
            class_name=str(optional_value(model, "class_name", "ConvAutoEncoder")),
            latent_dim=as_int(require_value(model, "model", "latent_dim"), "model.latent_dim"),
            kwargs=normalize_model_kwargs(model),
        ),
        optimizer=OptimizerConfig(
            name=optimizer_name,
            module=str(
                optional_value(
                    optimizer,
                    "module",
                    "keyframe_pipeline.optimizers.torch_optimizers",
                )
            ),
            class_name=str(
                optional_value(
                    optimizer,
                    "class_name",
                    default_optimizer_class_name(optimizer_name),
                )
            ),
            learning_rate=as_float(
                optional_value(optimizer, "learning_rate", legacy_learning_rate),
                "optimizer.learning_rate",
            ),
            weight_decay=as_float(optional_value(optimizer, "weight_decay", 0.0), "optimizer.weight_decay"),
            momentum=as_float(optional_value(optimizer, "momentum", 0.0), "optimizer.momentum"),
            kwargs=as_kwargs(optional_value(optimizer, "kwargs", {}), "optimizer.kwargs"),
        ),
        loss=LossConfig(
            name=loss_name,
            module=str(optional_value(loss, "module", "keyframe_pipeline.losses.basic")),
            class_name=str(
                optional_value(
                    loss,
                    "class_name",
                    default_loss_class_name(loss_name),
                )
            ),
            kwargs=as_kwargs(optional_value(loss, "kwargs", {}), "loss.kwargs"),
        ),
        train=TrainConfig(
            epochs=as_int(require_value(train, "train", "epochs"), "train.epochs"),
            batch_size=as_int(require_value(train, "train", "batch_size"), "train.batch_size"),
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
            name=visualization_name,
            module=str(
                optional_value(
                    visualization,
                    "module",
                    "keyframe_pipeline.visualizers.plotly_latent",
                )
            ),
            class_name=str(
                optional_value(
                    visualization,
                    "class_name",
                    default_visualization_class_name(visualization_name),
                )
            ),
            kwargs=as_kwargs(optional_value(visualization, "kwargs", {}), "visualization.kwargs"),
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
    if not config.model.module:
        raise ValueError("model.module 값이 필요합니다.")
    if not config.model.class_name:
        raise ValueError("model.class_name 값이 필요합니다.")
    if "encoder_channels" in config.model.kwargs and any(
        channel <= 0 for channel in config.model.kwargs["encoder_channels"]
    ):
        raise ValueError("model.kwargs.encoder_channels의 모든 값은 1 이상이어야 합니다.")
    if config.train.epochs <= 0:
        raise ValueError("train.epochs는 1 이상이어야 합니다.")
    if config.train.batch_size <= 0:
        raise ValueError("train.batch_size는 1 이상이어야 합니다.")
    if config.train.num_workers < 0:
        raise ValueError("train.num_workers는 0 이상이어야 합니다.")
    if config.train.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("train.device는 'auto', 'cpu', 'cuda' 중 하나여야 합니다.")
    if config.optimizer.learning_rate <= 0:
        raise ValueError("optimizer.learning_rate는 0보다 커야 합니다.")
    if config.optimizer.weight_decay < 0:
        raise ValueError("optimizer.weight_decay는 0 이상이어야 합니다.")
    if config.optimizer.momentum < 0:
        raise ValueError("optimizer.momentum은 0 이상이어야 합니다.")
    if not config.optimizer.module:
        raise ValueError("optimizer.module 값이 필요합니다.")
    if not config.optimizer.class_name:
        raise ValueError("optimizer.class_name 값이 필요합니다.")
    if not config.loss.module:
        raise ValueError("loss.module 값이 필요합니다.")
    if not config.loss.class_name:
        raise ValueError("loss.class_name 값이 필요합니다.")
    if not config.selection.module:
        raise ValueError("selection.module 값이 필요합니다.")
    if not config.selection.class_name:
        raise ValueError("selection.class_name 값이 필요합니다.")
    if config.visualization.dimensions not in {2, 3}:
        raise ValueError("visualization.dimensions는 2 또는 3이어야 합니다.")
    if config.visualization.annotate_every < 0:
        raise ValueError("visualization.annotate_every는 0 이상이어야 합니다.")
    if not config.visualization.module:
        raise ValueError("visualization.module 값이 필요합니다.")
    if not config.visualization.class_name:
        raise ValueError("visualization.class_name 값이 필요합니다.")
