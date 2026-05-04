from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch.nn.functional as F


class ConvAutoEncoder3D(nn.Module):
    def __init__(self, in_channels: int, input_height: int, input_width: int, latent_dim: int = 3) -> None:
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


def compute_downsampled_size(size: int, layers: int) -> int:
    value = size
    for _ in range(layers):
        value = ((value + 2 - 4) // 2) + 1
    if value <= 0:
        raise ValueError("입력 프레임 크기가 너무 작아서 4단계 다운샘플링을 처리할 수 없습니다.")
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D latent autoencoder를 학습합니다.")
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
        "--plot-dir",
        type=Path,
        default=Path("artifacts/plots"),
        help="학습 중간 시각화 저장 디렉토리",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="실험 이름. 비우면 dataset 상위 디렉토리명을 사용합니다.",
    )
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=0,
        help="0보다 크면 N epoch마다 3D latent snapshot을 HTML로 저장합니다.",
    )
    parser.add_argument(
        "--annotate-every",
        type=int,
        default=0,
        help="snapshot 시각화에서 N 간격으로 프레임 번호를 표시합니다. 0이면 표시하지 않습니다.",
    )
    return parser.parse_args()


def ensure_plotly_available() -> tuple[object, object]:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise RuntimeError(
            "3D latent HTML 시각화를 사용하려면 plotly가 필요합니다. requirements.txt를 다시 설치하세요."
        ) from exc
    return go, pio


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")

    with np.load(dataset_path) as dataset:
        train_frames = (
            dataset["train_frames"].astype(np.float32)
            if "train_frames" in dataset
            else dataset["frames"].astype(np.float32)
        )
        train_frame_indices = (
            dataset["train_frame_indices"].astype(np.int32)
            if "train_frame_indices" in dataset
            else dataset["frame_indices"].astype(np.int32)
        )
        train_timestamps_sec = (
            dataset["train_timestamps_sec"].astype(np.float32)
            if "train_timestamps_sec" in dataset
            else dataset["timestamps_sec"].astype(np.float32)
        )
        val_frames = dataset["val_frames"].astype(np.float32) if "val_frames" in dataset else None
        val_frame_indices = (
            dataset["val_frame_indices"].astype(np.int32) if "val_frame_indices" in dataset else None
        )
        val_timestamps_sec = (
            dataset["val_timestamps_sec"].astype(np.float32) if "val_timestamps_sec" in dataset else None
        )

    if train_frames.ndim != 4:
        raise ValueError("train_frames 배열은 (N, C, H, W) 형태여야 합니다.")
    if val_frames is not None and val_frames.ndim != 4:
        raise ValueError("val_frames 배열은 (N, C, H, W) 형태여야 합니다.")

    return (
        train_frames,
        train_frame_indices,
        train_timestamps_sec,
        val_frames,
        val_frame_indices,
        val_timestamps_sec,
    )


def evaluate_autoencoder(
    model: ConvAutoEncoder3D,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            batch_size = batch.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size

    return total_loss / max(sample_count, 1)


def encode_all_frames(
    model: ConvAutoEncoder3D,
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


def build_interactive_figure(
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    title: str,
    annotate_every: int,
) -> object:
    go, _ = ensure_plotly_available()
    order = np.arange(len(latents))
    custom_data = np.stack([order, frame_indices, timestamps_sec], axis=1)
    start_data = np.array([[order[0], frame_indices[0], timestamps_sec[0]]], dtype=np.float32)
    end_data = np.array([[order[-1], frame_indices[-1], timestamps_sec[-1]]], dtype=np.float32)

    marker_trace = go.Scatter3d(
        x=latents[:, 0],
        y=latents[:, 1],
        z=latents[:, 2],
        mode="markers",
        name="frames",
        marker={
            "size": 2,
            "color": order,
            "colorscale": "Viridis",
            "opacity": 0.58,
            "colorbar": {"title": "Frame Order"},
        },
        customdata=custom_data,
        hovertemplate=(
            "order=%{customdata[0]}<br>"
            "frame_idx=%{customdata[1]}<br>"
            "time=%{customdata[2]:.3f}s<br>"
            "z1=%{x:.4f}<br>z2=%{y:.4f}<br>z3=%{z:.4f}<extra></extra>"
        ),
    )
    path_shadow_trace = go.Scatter3d(
        x=latents[:, 0],
        y=latents[:, 1],
        z=latents[:, 2],
        mode="lines",
        name="trajectory-shadow",
        line={"color": "rgba(15, 23, 42, 0.18)", "width": 10},
        hoverinfo="skip",
        showlegend=False,
    )
    line_trace = go.Scatter3d(
        x=latents[:, 0],
        y=latents[:, 1],
        z=latents[:, 2],
        mode="lines",
        name="trajectory",
        line={"color": "rgba(37, 99, 235, 0.96)", "width": 5},
        hoverinfo="skip",
    )
    start_trace = go.Scatter3d(
        x=[latents[0, 0]],
        y=[latents[0, 1]],
        z=[latents[0, 2]],
        mode="markers+text",
        name="start",
        marker={
            "size": 8,
            "color": "#16a34a",
            "symbol": "circle",
            "line": {"color": "#052e16", "width": 5},
        },
        text=["START"],
        textposition="top center",
        textfont={"size": 12, "color": "#166534"},
        customdata=start_data,
        hovertemplate=(
            "start<br>"
            "order=%{customdata[0]}<br>"
            "frame_idx=%{customdata[1]}<br>"
            "time=%{customdata[2]:.3f}s<extra></extra>"
        ),
    )
    end_trace = go.Scatter3d(
        x=[latents[-1, 0]],
        y=[latents[-1, 1]],
        z=[latents[-1, 2]],
        mode="markers+text",
        name="end",
        marker={
            "size": 9,
            "color": "#dc2626",
            "symbol": "diamond",
            "line": {"color": "#7f1d1d", "width": 5},
        },
        text=["END"],
        textposition="top center",
        textfont={"size": 12, "color": "#991b1b"},
        customdata=end_data,
        hovertemplate=(
            "end<br>"
            "order=%{customdata[0]}<br>"
            "frame_idx=%{customdata[1]}<br>"
            "time=%{customdata[2]:.3f}s<extra></extra>"
        ),
    )

    traces = [path_shadow_trace, line_trace, marker_trace, start_trace, end_trace]
    if annotate_every > 0:
        annotated_indices = np.arange(0, len(latents), annotate_every)
        text_trace = go.Scatter3d(
            x=latents[annotated_indices, 0],
            y=latents[annotated_indices, 1],
            z=latents[annotated_indices, 2],
            mode="text",
            name="labels",
            text=[str(index) for index in annotated_indices],
            textposition="top center",
            hoverinfo="skip",
        )
        traces.append(text_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        template="plotly_white",
        scene={
            "xaxis_title": "Latent Dimension 1",
            "yaxis_title": "Latent Dimension 2",
            "zaxis_title": "Latent Dimension 3",
            "aspectmode": "data",
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def save_interactive_plot(
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    output_path: Path,
    title: str,
    annotate_every: int,
) -> None:
    _, pio = ensure_plotly_available()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_interactive_figure(
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        title=title,
        annotate_every=annotate_every,
    )
    pio.write_html(fig, file=str(output_path), auto_open=False, include_plotlyjs=True, full_html=True)


def save_training_snapshot(
    latents: np.ndarray,
    history: list[float],
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    dataset_path: Path,
    model_basename: str,
    epoch: int,
    latent_dir: Path,
    plot_dir: Path,
    annotate_every: int,
) -> tuple[Path, Path]:
    snapshot_latent_dir = latent_dir / f"{model_basename}_training"
    snapshot_plot_dir = plot_dir / f"{model_basename}_training"
    snapshot_latent_dir.mkdir(parents=True, exist_ok=True)
    snapshot_plot_dir.mkdir(parents=True, exist_ok=True)

    latent_snapshot_path = snapshot_latent_dir / f"epoch_{epoch:03d}_latents.npz"
    np.savez_compressed(
        latent_snapshot_path,
        latents=latents.astype(np.float32),
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        dataset_path=np.array(str(dataset_path.resolve())),
        epoch=np.array(epoch, dtype=np.int32),
        train_loss=np.array(history[-1] if history else np.nan, dtype=np.float32),
    )

    plot_snapshot_path = snapshot_plot_dir / f"epoch_{epoch:03d}.html"
    save_interactive_plot(
        latents=latents,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        output_path=plot_snapshot_path,
        title=f"{model_basename} Epoch {epoch:03d}",
        annotate_every=annotate_every,
    )
    return latent_snapshot_path, plot_snapshot_path


def train_autoencoder(
    model: ConvAutoEncoder3D,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    epoch_end_callback: Callable[[int, ConvAutoEncoder3D, list[float], list[float] | None], None] | None = None,
) -> tuple[list[float], list[float] | None]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_history: list[float] = []
    val_history: list[float] | None = [] if val_dataloader is not None else None

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for (batch,) in train_dataloader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch.size(0)
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = epoch_loss / max(sample_count, 1)
        train_history.append(float(train_loss))

        if val_dataloader is not None and val_history is not None:
            val_loss = evaluate_autoencoder(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device,
            )
            val_history.append(float(val_loss))
            print(
                f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )
        else:
            print(f"[Epoch {epoch + 1:03d}/{epochs:03d}] train_loss={train_loss:.6f}")

        if epoch_end_callback is not None:
            epoch_end_callback(epoch + 1, model, train_history, val_history)

    return train_history, val_history


def save_training_outputs(
    model: ConvAutoEncoder3D,
    train_history: list[float],
    val_history: list[float] | None,
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    split_labels: np.ndarray | None,
    dataset_path: Path,
    run_name: str,
    model_dir: Path,
    latent_dir: Path,
    plot_dir: Path,
) -> tuple[Path, Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model_basename = f"{run_name}_3d"
    checkpoint_path = model_dir / f"{model_basename}.pt"
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
            "train_history": train_history,
            "validation_history": val_history,
        },
        checkpoint_path,
    )

    history_path = model_dir / f"{model_basename}_history.json"
    history_payload = {
        "run_name": run_name,
        "dataset_path": str(dataset_path.resolve()),
        "epochs": len(train_history),
        "train_loss": train_history,
        "validation_loss": val_history,
        "final_train_loss": train_history[-1] if train_history else None,
        "final_validation_loss": val_history[-1] if val_history else None,
        "best_validation_loss": min(val_history) if val_history else None,
    }
    history_path.write_text(json.dumps(history_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    loss_plot_path = plot_dir / f"{model_basename}_loss.png"
    save_loss_curve(
        train_history=train_history,
        validation_history=val_history,
        output_path=loss_plot_path,
        title=f"Loss Curve: {model_basename}",
    )

    latent_path = latent_dir / f"{model_basename}_latents.npz"
    latent_payload = {
        "latents": latents.astype(np.float32),
        "frame_indices": frame_indices,
        "timestamps_sec": timestamps_sec,
        "dataset_path": np.array(str(dataset_path.resolve())),
        "checkpoint_path": np.array(str(checkpoint_path.resolve())),
        "run_name": np.array(run_name),
        "latent_dim": np.array(3, dtype=np.int32),
        "final_loss": np.array(train_history[-1] if train_history else np.nan, dtype=np.float32),
    }
    if split_labels is not None:
        latent_payload["split_labels"] = split_labels.astype(np.int32)
    np.savez_compressed(latent_path, **latent_payload)
    return checkpoint_path, latent_path, loss_plot_path


def save_loss_curve(
    train_history: list[float],
    validation_history: list[float] | None,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_history) + 1, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_history, color="#2563eb", linewidth=2.0, label="train loss")
    if validation_history:
        ax.plot(epochs, validation_history, color="#dc2626", linewidth=2.0, label="validation loss")
    if train_history:
        ax.scatter([epochs[0]], [train_history[0]], color="#16a34a", s=50, zorder=3, label="train start")
        ax.scatter(
            [epochs[-1]],
            [train_history[-1]],
            color="#1d4ed8",
            s=50,
            zorder=3,
            label="train end",
        )
    if validation_history:
        ax.scatter(
            [epochs[-1]],
            [validation_history[-1]],
            color="#991b1b",
            s=50,
            zorder=3,
            label="validation end",
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.visualize_every < 0:
        raise ValueError("--visualize-every 값은 0 이상이어야 합니다.")
    if args.annotate_every < 0:
        raise ValueError("--annotate-every 값은 0 이상이어야 합니다.")
    if args.visualize_every > 0:
        ensure_plotly_available()

    (
        train_frames,
        train_frame_indices,
        train_timestamps_sec,
        val_frames,
        val_frame_indices,
        val_timestamps_sec,
    ) = load_dataset(args.dataset)
    run_name = args.run_name or args.dataset.parent.name
    model_basename = f"{run_name}_3d"

    train_tensor_frames = torch.from_numpy(train_frames)
    train_dataloader = DataLoader(
        TensorDataset(train_tensor_frames),
        batch_size=min(args.batch_size, len(train_tensor_frames)),
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = None
    if val_frames is not None:
        val_tensor_frames = torch.from_numpy(val_frames)
        val_dataloader = DataLoader(
            TensorDataset(val_tensor_frames),
            batch_size=min(args.batch_size, len(val_tensor_frames)),
            shuffle=False,
            num_workers=args.num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    model = ConvAutoEncoder3D(
        in_channels=train_frames.shape[1],
        input_height=train_frames.shape[-2],
        input_width=train_frames.shape[-1],
        latent_dim=3,
    )

    def on_epoch_end(
        epoch: int,
        trained_model: ConvAutoEncoder3D,
        train_history: list[float],
        val_history: list[float] | None,
    ) -> None:
        if args.visualize_every == 0:
            return
        if epoch == args.epochs:
            return
        if epoch % args.visualize_every != 0:
            return

        latents = encode_all_frames(
            model=trained_model,
            frames=train_frames,
            batch_size=min(args.batch_size, len(train_frames)),
            device=device,
        )
        _, snapshot_plot_path = save_training_snapshot(
            latents=latents,
            history=train_history,
            frame_indices=train_frame_indices,
            timestamps_sec=train_timestamps_sec,
            dataset_path=args.dataset,
            model_basename=model_basename,
            epoch=epoch,
            latent_dir=args.latent_dir,
            plot_dir=args.plot_dir,
            annotate_every=args.annotate_every,
        )
        print(f"[Epoch {epoch:03d}] snapshot 저장 경로: {snapshot_plot_path}")

    train_history, val_history = train_autoencoder(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        epoch_end_callback=on_epoch_end,
    )
    train_latents = encode_all_frames(
        model=model,
        frames=train_frames,
        batch_size=min(args.batch_size, len(train_frames)),
        device=device,
    )
    combined_latents = train_latents
    combined_frame_indices = train_frame_indices
    combined_timestamps_sec = train_timestamps_sec
    split_labels = np.zeros(len(train_latents), dtype=np.int32)
    if val_frames is not None and val_frame_indices is not None and val_timestamps_sec is not None:
        val_latents = encode_all_frames(
            model=model,
            frames=val_frames,
            batch_size=min(args.batch_size, len(val_frames)),
            device=device,
        )
        combined_latents = np.concatenate([train_latents, val_latents], axis=0)
        combined_frame_indices = np.concatenate([train_frame_indices, val_frame_indices], axis=0)
        combined_timestamps_sec = np.concatenate([train_timestamps_sec, val_timestamps_sec], axis=0)
        split_labels = np.concatenate(
            [
                np.zeros(len(train_latents), dtype=np.int32),
                np.ones(len(val_latents), dtype=np.int32),
            ],
            axis=0,
        )
        order = np.argsort(combined_frame_indices)
        combined_latents = combined_latents[order]
        combined_frame_indices = combined_frame_indices[order]
        combined_timestamps_sec = combined_timestamps_sec[order]
        split_labels = split_labels[order]

    checkpoint_path, latent_path, loss_plot_path = save_training_outputs(
        model=model,
        train_history=train_history,
        val_history=val_history,
        latents=combined_latents,
        frame_indices=combined_frame_indices,
        timestamps_sec=combined_timestamps_sec,
        split_labels=split_labels,
        dataset_path=args.dataset,
        run_name=run_name,
        model_dir=args.model_dir,
        latent_dir=args.latent_dir,
        plot_dir=args.plot_dir,
    )

    if args.visualize_every > 0:
        _, final_plot_path = save_training_snapshot(
            latents=combined_latents,
            history=train_history,
            frame_indices=combined_frame_indices,
            timestamps_sec=combined_timestamps_sec,
            dataset_path=args.dataset,
            model_basename=model_basename,
            epoch=args.epochs,
            latent_dir=args.latent_dir,
            plot_dir=args.plot_dir,
            annotate_every=args.annotate_every,
        )
        print(f"최종 3D 시각화 저장 경로: {final_plot_path}")

    print(f"체크포인트 저장 경로: {checkpoint_path}")
    print(f"latent 저장 경로: {latent_path}")
    print(f"loss 그래프 저장 경로: {loss_plot_path}")


if __name__ == "__main__":
    main()
