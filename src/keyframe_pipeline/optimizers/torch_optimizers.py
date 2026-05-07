from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from keyframe_pipeline.config import OptimizerConfig


class OptimizerStrategy(ABC):
    name: str

    @abstractmethod
    def build(self, model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
        raise NotImplementedError


class AdamOptimizerStrategy(OptimizerStrategy):
    name = "adam"

    def build(self, model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **config.kwargs,
        )


class AdamWOptimizerStrategy(OptimizerStrategy):
    name = "adamw"

    def build(self, model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **config.kwargs,
        )


class SGDOptimizerStrategy(OptimizerStrategy):
    name = "sgd"

    def build(self, model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            **config.kwargs,
        )


OPTIMIZER_STRATEGIES: dict[str, OptimizerStrategy] = {
    strategy.name: strategy
    for strategy in (
        AdamOptimizerStrategy(),
        AdamWOptimizerStrategy(),
        SGDOptimizerStrategy(),
    )
}


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    try:
        strategy = OPTIMIZER_STRATEGIES[config.name]
    except KeyError as exc:
        available = ", ".join(sorted(OPTIMIZER_STRATEGIES))
        raise ValueError(f"지원하지 않는 optimizer 전략입니다: {config.name}. available={available}") from exc
    return strategy.build(model=model, config=config)
