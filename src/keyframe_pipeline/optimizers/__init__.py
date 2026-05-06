from __future__ import annotations

import torch
from torch import nn

from keyframe_pipeline.config import OptimizerConfig
from keyframe_pipeline.loading import instantiate_class, require_method


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    strategy = instantiate_class(module_path=config.module, class_name=config.class_name)
    require_method(strategy, "build", f"{config.module}.{config.class_name}")
    optimizer = strategy.build(model=model, config=config)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            f"{config.module}.{config.class_name}.build()는 torch.optim.Optimizer를 반환해야 합니다."
        )
    return optimizer

