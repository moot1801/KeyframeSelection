from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from keyframe_pipeline.config import PipelineConfig
from keyframe_pipeline.models import checkpoint_model_config


def write_selected_csv(
    output_path: Path,
    selected_candidate_orders: np.ndarray,
    selected_frame_indices: np.ndarray,
    selected_timestamps_sec: np.ndarray,
    cumulative: np.ndarray,
    distances: np.ndarray,
    image_paths: list[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_distances = np.concatenate([[np.nan], distances]).astype(np.float32)
    next_distances = np.concatenate([distances, [np.nan]]).astype(np.float32)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "selection_order",
                "candidate_order",
                "frame_index",
                "timestamp_sec",
                "cumulative_latent_distance",
                "prev_latent_distance",
                "next_latent_distance",
                "image_path",
            ],
        )
        writer.writeheader()
        for order, candidate_order in enumerate(selected_candidate_orders):
            writer.writerow(
                {
                    "selection_order": order,
                    "candidate_order": int(candidate_order),
                    "frame_index": int(selected_frame_indices[order]),
                    "timestamp_sec": f"{float(selected_timestamps_sec[order]):.6f}",
                    "cumulative_latent_distance": f"{float(cumulative[candidate_order]):.6f}",
                    "prev_latent_distance": (
                        ""
                        if np.isnan(previous_distances[order])
                        else f"{float(previous_distances[order]):.6f}"
                    ),
                    "next_latent_distance": (
                        ""
                        if np.isnan(next_distances[order])
                        else f"{float(next_distances[order]):.6f}"
                    ),
                    "image_path": str(image_paths[order]),
                }
            )


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    history: list[float],
    config: PipelineConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": checkpoint_model_config(model, config=config.model),
            "train_loss": history,
            "source_video": str(config.video.input_path),
            "optimizer": {
                "name": config.optimizer.name,
                "module": config.optimizer.module,
                "class_name": config.optimizer.class_name,
                "learning_rate": config.optimizer.learning_rate,
                "weight_decay": config.optimizer.weight_decay,
                "momentum": config.optimizer.momentum,
                "kwargs": config.optimizer.kwargs,
            },
            "loss": {
                "name": config.loss.name,
                "module": config.loss.module,
                "class_name": config.loss.class_name,
                "kwargs": config.loss.kwargs,
            },
        },
        output_path,
    )


def save_latent_npz(
    output_path: Path,
    latents: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    selected_candidate_orders: np.ndarray,
    distances: np.ndarray,
    fps: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        latents=latents.astype(np.float32),
        frame_indices=frame_indices.astype(np.int32),
        timestamps_sec=timestamps_sec.astype(np.float32),
        selected_candidate_orders=selected_candidate_orders.astype(np.int32),
        selected_frame_indices=frame_indices[selected_candidate_orders].astype(np.int32),
        selected_distances=distances.astype(np.float32),
        fps=np.array(fps, dtype=np.float32),
    )


def save_metrics_json(
    output_path: Path,
    history: list[float],
    selection_result: object,
    uniform_frame_indices: np.ndarray,
    frame_indices: np.ndarray,
    config: PipelineConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_distances = selection_result.initial_distances
    final_distances = selection_result.final_distances
    payload = {
        "source_video": str(config.video.input_path),
        "training_candidate_num_frames": config.video.candidate_num_frames,
        "selection_source_frame_count": int(len(frame_indices)),
        "selected_num_frames": config.selection.num_frames,
        "initial_selected_frame_indices": frame_indices[selection_result.initial_selected].astype(int).tolist(),
        "final_selected_frame_indices": frame_indices[selection_result.final_selected].astype(int).tolist(),
        "uniform_frame_indices": uniform_frame_indices.astype(int).tolist(),
        "initial_distance_mean": float(np.mean(initial_distances)) if len(initial_distances) else 0.0,
        "initial_distance_variance": float(np.var(initial_distances)) if len(initial_distances) else 0.0,
        "final_distance_mean": float(np.mean(final_distances)) if len(final_distances) else 0.0,
        "final_distance_variance": float(np.var(final_distances)) if len(final_distances) else 0.0,
        "final_distance_std": float(np.std(final_distances)) if len(final_distances) else 0.0,
        "final_distances": final_distances.astype(float).tolist(),
        "epochs": config.train.epochs,
        "final_train_loss": history[-1] if history else None,
        "strategies": {
            "selection": {
                "name": config.selection.name,
                "module": config.selection.module,
                "class_name": config.selection.class_name,
            },
            "model": {
                "name": config.model.name,
                "module": config.model.module,
                "class_name": config.model.class_name,
            },
            "optimizer": {
                "name": config.optimizer.name,
                "module": config.optimizer.module,
                "class_name": config.optimizer.class_name,
            },
            "loss": {
                "name": config.loss.name,
                "module": config.loss.module,
                "class_name": config.loss.class_name,
            },
            "visualization": {
                "name": config.visualization.name,
                "module": config.visualization.module,
                "class_name": config.visualization.class_name,
            },
        },
        "latent_dim": config.model.latent_dim,
        "selection_method": "all_video_frame_latent_arclength_with_local_variance_refinement",
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
