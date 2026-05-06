from __future__ import annotations

from typing import Any

from torch import nn

from keyframe_pipeline.config import ModelConfig
from keyframe_pipeline.loading import load_class


def build_model(
    config: ModelConfig,
    in_channels: int,
    input_height: int,
    input_width: int,
) -> nn.Module:
    model_class = load_class(module_path=config.module, class_name=config.class_name)
    kwargs: dict[str, Any] = dict(config.kwargs)
    kwargs.setdefault("in_channels", in_channels)
    kwargs.setdefault("input_height", input_height)
    kwargs.setdefault("input_width", input_width)
    kwargs.setdefault("latent_dim", config.latent_dim)

    try:
        model = model_class(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{config.module}.{config.class_name} 생성에 실패했습니다. "
            "모델 __init__ 인자와 model.kwargs를 확인하세요."
        ) from exc

    if not isinstance(model, nn.Module):
        raise TypeError(f"{config.module}.{config.class_name}은 torch.nn.Module이어야 합니다.")
    encode = getattr(model, "encode", None)
    if not callable(encode):
        raise TypeError("모델은 전체 영상 latent 추출을 위해 encode(inputs) 메서드를 구현해야 합니다.")
    return model


def checkpoint_model_config(model: nn.Module, config: ModelConfig) -> dict[str, object]:
    metadata: dict[str, object] = {
        "name": config.name,
        "module": config.module,
        "class_name": config.class_name,
        "latent_dim": config.latent_dim,
        "kwargs": config.kwargs,
    }
    for attr_name in ("in_channels", "input_height", "input_width"):
        if hasattr(model, attr_name):
            metadata[attr_name] = getattr(model, attr_name)
    return metadata

