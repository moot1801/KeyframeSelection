from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def compute_downsampled_size(size: int, layers: int) -> int:
    value = size
    for _ in range(layers):
        value = ((value + 2 - 4) // 2) + 1
    if value <= 0:
        raise ValueError("입력 프레임 크기가 너무 작아서 다운샘플링을 처리할 수 없습니다.")
    return int(value)


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"지원하지 않는 activation입니다: {name}")


def build_output_activation(name: str) -> nn.Module:
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"지원하지 않는 output_activation입니다: {name}")


class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_height: int,
        input_width: int,
        latent_dim: int,
        encoder_channels: tuple[int, ...] = (32, 64, 128, 256),
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder_channels = encoder_channels
        self.activation = activation
        self.output_activation = output_activation
        self.feature_map_height = compute_downsampled_size(input_height, layers=len(encoder_channels))
        self.feature_map_width = compute_downsampled_size(input_width, layers=len(encoder_channels))

        encoder_layers: list[nn.Module] = []
        current_channels = in_channels
        for next_channels in encoder_channels:
            encoder_layers.extend(
                [
                    nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
                    build_activation(activation),
                ]
            )
            current_channels = next_channels
        self.encoder = nn.Sequential(*encoder_layers)

        flattened_size = current_channels * self.feature_map_height * self.feature_map_width
        self.to_latent = nn.Linear(flattened_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened_size)

        decoder_layers: list[nn.Module] = []
        decoder_channels = list(reversed(encoder_channels))
        current_channels = decoder_channels[0]
        for next_channels in decoder_channels[1:]:
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
                    build_activation(activation),
                ]
            )
            current_channels = next_channels
        decoder_layers.append(
            nn.ConvTranspose2d(current_channels, in_channels, kernel_size=4, stride=2, padding=1)
        )
        decoder_layers.append(build_output_activation(output_activation))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        flattened = features.flatten(start_dim=1)
        return self.to_latent(flattened)

    def decode(self, latent: torch.Tensor, output_hw: tuple[int, int] | None = None) -> torch.Tensor:
        restored = self.from_latent(latent)
        restored = restored.view(
            -1,
            self.encoder_channels[-1],
            self.feature_map_height,
            self.feature_map_width,
        )
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

