from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from keyframe_pipeline.clustering import compute_selected_clusters, cluster_summaries
from keyframe_pipeline.config import PipelineConfig
from keyframe_pipeline.models import checkpoint_model_config


def write_selected_csv(
    output_path: Path,
    selected_candidate_orders: np.ndarray,
    selected_frame_indices: np.ndarray,
    selected_timestamps_sec: np.ndarray,
    cumulative: np.ndarray,
    distances: np.ndarray,
    latents: np.ndarray | None,
    image_paths: list[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_distances = np.concatenate([[np.nan], distances]).astype(np.float32)
    next_distances = np.concatenate([distances, [np.nan]]).astype(np.float32)
    cluster_info = compute_selected_clusters(
        distances,
        latents=latents,
        selected_candidate_orders=selected_candidate_orders,
    )

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
                "cluster_id",
                "cluster_start",
                "next_cluster_boundary",
                "image_path",
            ],
        )
        writer.writeheader()
        for order, candidate_order in enumerate(selected_candidate_orders):
            cluster_id = int(cluster_info.cluster_ids[order])
            cluster_start = order == 0 or cluster_id != int(cluster_info.cluster_ids[order - 1])
            next_cluster_boundary = (
                order < len(cluster_info.boundary_mask) and bool(cluster_info.boundary_mask[order])
            )
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
                    "cluster_id": cluster_id,
                    "cluster_start": int(cluster_start),
                    "next_cluster_boundary": int(next_cluster_boundary),
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
    cluster_info = compute_selected_clusters(
        distances,
        latents=latents,
        selected_candidate_orders=selected_candidate_orders,
    )
    np.savez_compressed(
        output_path,
        latents=latents.astype(np.float32),
        frame_indices=frame_indices.astype(np.int32),
        timestamps_sec=timestamps_sec.astype(np.float32),
        selected_candidate_orders=selected_candidate_orders.astype(np.int32),
        selected_frame_indices=frame_indices[selected_candidate_orders].astype(np.int32),
        selected_distances=distances.astype(np.float32),
        selected_cluster_ids=cluster_info.cluster_ids.astype(np.int32),
        selected_cluster_boundary_mask=cluster_info.boundary_mask.astype(np.bool_),
        selected_cluster_selected_boundary_mask=cluster_info.selected_boundary_mask.astype(np.bool_),
        selected_cluster_candidate_boundary_mask=cluster_info.candidate_boundary_mask.astype(np.bool_),
        selected_cluster_threshold=np.array(cluster_info.threshold, dtype=np.float32),
        selected_cluster_local_medians=cluster_info.local_medians.astype(np.float32),
        selected_cluster_local_mads=cluster_info.local_mads.astype(np.float32),
        selected_cluster_local_scales=cluster_info.local_scales.astype(np.float32),
        selected_cluster_local_thresholds=cluster_info.local_thresholds.astype(np.float32),
        selected_cluster_ratio_thresholds=cluster_info.ratio_thresholds.astype(np.float32),
        selected_cluster_effective_thresholds=cluster_info.effective_thresholds.astype(np.float32),
        selected_cluster_candidate_gaps=cluster_info.candidate_gaps.astype(np.int32),
        selected_cluster_candidate_context_counts=cluster_info.candidate_context_counts.astype(np.int32),
        selected_cluster_candidate_local_medians=cluster_info.candidate_local_medians.astype(np.float32),
        selected_cluster_candidate_local_mads=cluster_info.candidate_local_mads.astype(np.float32),
        selected_cluster_candidate_local_scales=cluster_info.candidate_local_scales.astype(np.float32),
        selected_cluster_candidate_local_thresholds=cluster_info.candidate_local_thresholds.astype(np.float32),
        selected_cluster_candidate_expected_distances=cluster_info.candidate_expected_distances.astype(np.float32),
        selected_cluster_candidate_distance_ratios=cluster_info.candidate_distance_ratios.astype(np.float32),
        fps=np.array(fps, dtype=np.float32),
    )


def save_metrics_json(
    output_path: Path,
    history: list[float],
    selection_result: object,
    uniform_frame_indices: np.ndarray,
    frame_indices: np.ndarray,
    latents: np.ndarray,
    config: PipelineConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_distances = selection_result.initial_distances
    final_distances = selection_result.final_distances
    final_selected_frame_indices = frame_indices[selection_result.final_selected].astype(int)
    cluster_info = compute_selected_clusters(
        final_distances,
        latents=latents,
        selected_candidate_orders=selection_result.final_selected,
    )
    cluster_boundaries = [
        {
            "distance_order": int(order),
            "from_selection_order": int(order),
            "to_selection_order": int(order + 1),
            "from_frame_index": int(final_selected_frame_indices[order]),
            "to_frame_index": int(final_selected_frame_indices[order + 1]),
            "distance": float(final_distances[order]),
            "threshold": float(cluster_info.effective_thresholds[order]),
            "local_median": float(cluster_info.local_medians[order]),
            "local_mad": float(cluster_info.local_mads[order]),
            "local_scale": float(cluster_info.local_scales[order]),
            "local_threshold": float(cluster_info.local_thresholds[order]),
            "ratio_threshold": float(cluster_info.ratio_thresholds[order]),
            "global_threshold": float(cluster_info.threshold),
            "candidate_gap": int(cluster_info.candidate_gaps[order]),
            "candidate_context_count": int(cluster_info.candidate_context_counts[order]),
            "candidate_local_median": float(cluster_info.candidate_local_medians[order]),
            "candidate_local_mad": float(cluster_info.candidate_local_mads[order]),
            "candidate_local_scale": float(cluster_info.candidate_local_scales[order]),
            "candidate_local_threshold": float(cluster_info.candidate_local_thresholds[order]),
            "candidate_expected_distance": float(cluster_info.candidate_expected_distances[order]),
            "candidate_distance_ratio": float(cluster_info.candidate_distance_ratios[order]),
        }
        for order, is_boundary in enumerate(cluster_info.boundary_mask)
        if is_boundary
    ]
    payload = {
        "source_video": str(config.video.input_path),
        "training_candidate_num_frames": config.video.candidate_num_frames,
        "selection_source_frame_count": int(len(frame_indices)),
        "requested_num_frames": int(getattr(selection_result, "requested_num_frames", config.selection.num_frames)),
        "selected_num_frames": int(getattr(selection_result, "resolved_num_frames", len(selection_result.final_selected))),
        "actual_selected_num_frames": int(len(selection_result.final_selected)),
        "auto_num_frames": getattr(selection_result, "auto_num_frames_summary", None),
        "initial_selected_frame_indices": frame_indices[selection_result.initial_selected].astype(int).tolist(),
        "final_selected_frame_indices": final_selected_frame_indices.tolist(),
        "uniform_frame_indices": uniform_frame_indices.astype(int).tolist(),
        "initial_distance_mean": float(np.mean(initial_distances)) if len(initial_distances) else 0.0,
        "initial_distance_variance": float(np.var(initial_distances)) if len(initial_distances) else 0.0,
        "final_distance_mean": float(np.mean(final_distances)) if len(final_distances) else 0.0,
        "final_distance_variance": float(np.var(final_distances)) if len(final_distances) else 0.0,
        "final_distance_std": float(np.std(final_distances)) if len(final_distances) else 0.0,
        "final_distances": final_distances.astype(float).tolist(),
        "selected_cluster_method": "robust_selected_with_candidate_context_density",
        "selected_cluster_threshold": float(cluster_info.threshold),
        "selected_cluster_global_threshold": float(cluster_info.threshold),
        "selected_cluster_local_window": int(cluster_info.local_window),
        "selected_cluster_local_mad_multiplier": float(cluster_info.local_mad_multiplier),
        "selected_cluster_global_percentile": float(cluster_info.global_percentile),
        "selected_cluster_min_distance_ratio": float(cluster_info.min_distance_ratio),
        "selected_cluster_min_cluster_size": int(cluster_info.min_cluster_size),
        "selected_cluster_candidate_context_window": int(cluster_info.candidate_context_window),
        "selected_cluster_candidate_mad_multiplier": float(cluster_info.candidate_mad_multiplier),
        "selected_cluster_candidate_distance_ratio": float(cluster_info.candidate_distance_ratio),
        "selected_cluster_distance_mean": float(cluster_info.distance_mean),
        "selected_cluster_distance_std": float(cluster_info.distance_std),
        "selected_cluster_distance_median": float(cluster_info.distance_median),
        "selected_cluster_distance_mad": float(cluster_info.distance_mad),
        "selected_cluster_local_medians": cluster_info.local_medians.astype(float).tolist(),
        "selected_cluster_local_mads": cluster_info.local_mads.astype(float).tolist(),
        "selected_cluster_local_scales": cluster_info.local_scales.astype(float).tolist(),
        "selected_cluster_local_thresholds": cluster_info.local_thresholds.astype(float).tolist(),
        "selected_cluster_ratio_thresholds": cluster_info.ratio_thresholds.astype(float).tolist(),
        "selected_cluster_effective_thresholds": cluster_info.effective_thresholds.astype(float).tolist(),
        "selected_cluster_candidate_gaps": cluster_info.candidate_gaps.astype(int).tolist(),
        "selected_cluster_candidate_context_counts": cluster_info.candidate_context_counts.astype(int).tolist(),
        "selected_cluster_candidate_local_medians": cluster_info.candidate_local_medians.astype(float).tolist(),
        "selected_cluster_candidate_local_mads": cluster_info.candidate_local_mads.astype(float).tolist(),
        "selected_cluster_candidate_local_scales": cluster_info.candidate_local_scales.astype(float).tolist(),
        "selected_cluster_candidate_local_thresholds": cluster_info.candidate_local_thresholds.astype(float).tolist(),
        "selected_cluster_candidate_expected_distances": cluster_info.candidate_expected_distances.astype(float).tolist(),
        "selected_cluster_candidate_distance_ratios": cluster_info.candidate_distance_ratios.astype(float).tolist(),
        "selected_cluster_selected_boundary_count": int(np.sum(cluster_info.selected_boundary_mask)),
        "selected_cluster_candidate_boundary_count": int(np.sum(cluster_info.candidate_boundary_mask)),
        "selected_cluster_boundary_count": int(np.sum(cluster_info.boundary_mask)),
        "selected_cluster_count": int(cluster_info.cluster_count),
        "selected_cluster_ids": cluster_info.cluster_ids.astype(int).tolist(),
        "selected_cluster_boundaries": cluster_boundaries,
        "selected_clusters": cluster_summaries(
            cluster_ids=cluster_info.cluster_ids,
            selected_candidate_orders=selection_result.final_selected,
            selected_frame_indices=final_selected_frame_indices,
        ),
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
