from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from keyframe_pipeline.config import LossConfig


class LossStrategy(ABC):
    name: str

    @abstractmethod
    def build(self, config: LossConfig) -> nn.Module:
        raise NotImplementedError


class MSELossStrategy(LossStrategy):
    name = "mse"

    def build(self, config: LossConfig) -> nn.Module:
        return nn.MSELoss(**config.kwargs)


class L1LossStrategy(LossStrategy):
    name = "l1"

    def build(self, config: LossConfig) -> nn.Module:
        return nn.L1Loss(**config.kwargs)


class SmoothL1LossStrategy(LossStrategy):
    name = "smooth_l1"

    def build(self, config: LossConfig) -> nn.Module:
        return nn.SmoothL1Loss(**config.kwargs)


LOSS_STRATEGIES: dict[str, LossStrategy] = {
    strategy.name: strategy
    for strategy in (
        MSELossStrategy(),
        L1LossStrategy(),
        SmoothL1LossStrategy(),
    )
}


def build_loss(config: LossConfig) -> nn.Module:
    try:
        strategy = LOSS_STRATEGIES[config.name]
    except KeyError as exc:
        available = ", ".join(sorted(LOSS_STRATEGIES))
        raise ValueError(f"지원하지 않는 loss 전략입니다: {config.name}. available={available}") from exc
    return strategy.build(config)
