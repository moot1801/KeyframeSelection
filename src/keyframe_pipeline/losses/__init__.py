from __future__ import annotations

from torch import nn

from keyframe_pipeline.config import LossConfig
from keyframe_pipeline.loading import instantiate_class, require_method


def build_loss(config: LossConfig) -> nn.Module:
    strategy = instantiate_class(module_path=config.module, class_name=config.class_name)
    require_method(strategy, "build", f"{config.module}.{config.class_name}")
    loss = strategy.build(config)
    if not isinstance(loss, nn.Module):
        raise TypeError(f"{config.module}.{config.class_name}.build()는 torch.nn.Module을 반환해야 합니다.")
    return loss

