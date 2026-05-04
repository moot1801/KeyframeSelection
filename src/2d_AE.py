from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def compute_downsampled_size(size: int, layers: int) -> int:
    value = size
    for _ in range(layers):
        value = ((value + 2 - 4) // 2) + 1
    if value <= 0:
        raise ValueError("입력 프레임 크기가 너무 작아서 4단계 다운샘플링을 처리할 수 없습니다.")
    return int(value)


class ConvAutoEncoder2D(nn.Module):
    def __init__(self, in_channels: int, input_height: int, input_width: int, latent_dim: int = 2) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D latent autoencoder를 학습합니다.")
    parser.add_argument("--dataset", type=Path, required=True, help="video_split.py가 생성한 dataset.npz 경로")
    parser.add_argument("--epochs", type=int, default=100, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=16, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="학습률")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 수")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="체크포인트 저장 디렉토리",
    )
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=Path("artifacts/latents"),
        help="latent 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="실험 이름. 비우면 dataset 상위 디렉토리명을 사용합니다.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")

    with np.load(dataset_path) as dataset:
        frames = dataset["frames"].astype(np.float32)
        frame_indices = dataset["frame_indices"].astype(np.int32)
        timestamps_sec = dataset["timestamps_sec"].astype(np.float32)

    if frames.ndim != 4:
        raise ValueError("frames 배열은 (N, C, H, W) 형태여야 합니다.")
    return frames, frame_indices, timestamps_sec


def train_autoencoder(
    model: ConvAutoEncoder2D,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: list[float] = []

    model.to(device)
    model.train()
    for epoch in range(epochs):
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
        print(f"[Epoch {epoch + 1:03d}/{epochs:03d}] loss={average_loss:.6f}")

    return history


def encode_all_frames(
    model: ConvAutoEncoder2D,
    frames: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor_frames = torch.from_numpy(frames)
    dataloader = DataLoader(TensorDataset(tensor_frames), batch_size=batch_size, shuffle=False)

    model.eval()
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)
            latent = model.encode(batch).cpu().numpy()
            latents.append(latent)

    return np.concatenate(latents, axis=0)


def save_training_outputs(
    model: ConvAutoEncoder2D,
    history: list[float],
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    dataset_path: Path,
    run_name: str,
    model_dir: Path,
    latent_dir: Path,
) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": model.in_channels,
                "input_height": model.input_height,
                "input_width": model.input_width,
                "latent_dim": model.latent_dim,
            },
            "dataset_path": str(dataset_path.resolve()),
            "history": history,
        },
        checkpoint_path,
    )

    history_path = model_dir / f"{run_name}_history.json"
    history_payload = {
        "run_name": run_name,
        "dataset_path": str(dataset_path.resolve()),
        "epochs": len(history),
        "train_loss": history,
        "final_loss": history[-1] if history else None,
    }
    history_path.write_text(json.dumps(history_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    latent_path = latent_dir / f"{run_name}_latents.npz"
    np.savez_compressed(
        latent_path,
        latents=latents.astype(np.float32),
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        dataset_path=np.array(str(dataset_path.resolve())),
        checkpoint_path=np.array(str(checkpoint_path.resolve())),
        run_name=np.array(run_name),
        final_loss=np.array(history[-1] if history else np.nan, dtype=np.float32),
    )
    return checkpoint_path, latent_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    frames, frame_indices, timestamps_sec = load_dataset(args.dataset)
    run_name = args.run_name or args.dataset.parent.name

    tensor_frames = torch.from_numpy(frames)
    dataloader = DataLoader(
        TensorDataset(tensor_frames),
        batch_size=min(args.batch_size, len(tensor_frames)),
        shuffle=True,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoEncoder2D(
        in_channels=frames.shape[1],
        input_height=frames.shape[-2],
        input_width=frames.shape[-1],
        latent_dim=2,
    )

    history = train_autoencoder(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    latents = encode_all_frames(
        model=model,
        frames=frames,
        batch_size=min(args.batch_size, len(frames)),
        device=device,
    )
    checkpoint_path, latent_path = save_training_outputs(
        model=model,
        history=history,
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        dataset_path=args.dataset,
        run_name=run_name,
        model_dir=args.model_dir,
        latent_dir=args.latent_dir,
    )

    print(f"체크포인트 저장 경로: {checkpoint_path}")
    print(f"latent 저장 경로: {latent_path}")


if __name__ == "__main__":
    main()
