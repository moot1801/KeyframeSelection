from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from keyframe_pipeline.config import LossConfig, OptimizerConfig, TrainConfig
from keyframe_pipeline.losses import build_loss
from keyframe_pipeline.optimizers import build_optimizer


def train_autoencoder(
    model: nn.Module,
    frames: np.ndarray,
    train_config: TrainConfig,
    optimizer_config: OptimizerConfig,
    loss_config: LossConfig,
    device: torch.device,
) -> list[float]:
    tensor_frames = torch.from_numpy(frames.astype(np.float32))
    dataloader = DataLoader(
        TensorDataset(tensor_frames),
        batch_size=min(train_config.batch_size, len(tensor_frames)),
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    model.to(device)
    optimizer = build_optimizer(model=model, config=optimizer_config)
    criterion = build_loss(loss_config)
    history: list[float] = []

    model.train()
    for epoch in range(train_config.epochs):
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
        print(f"[Epoch {epoch + 1:03d}/{train_config.epochs:03d}] loss={average_loss:.6f}")

    return history
