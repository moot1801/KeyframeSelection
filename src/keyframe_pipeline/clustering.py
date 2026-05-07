from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectedClusterInfo:
    distance_mean: float
    distance_std: float
    distance_median: float
    distance_mad: float
    threshold: float
    local_window: int
    local_mad_multiplier: float
    global_percentile: float
    min_distance_ratio: float
    min_cluster_size: int
    candidate_context_window: int
    candidate_mad_multiplier: float
    candidate_distance_ratio: float
    local_medians: np.ndarray
    local_mads: np.ndarray
    local_scales: np.ndarray
    local_thresholds: np.ndarray
    ratio_thresholds: np.ndarray
    effective_thresholds: np.ndarray
    selected_boundary_mask: np.ndarray
    candidate_gaps: np.ndarray
    candidate_context_counts: np.ndarray
    candidate_local_medians: np.ndarray
    candidate_local_mads: np.ndarray
    candidate_local_scales: np.ndarray
    candidate_local_thresholds: np.ndarray
    candidate_expected_distances: np.ndarray
    candidate_distance_ratios: np.ndarray
    candidate_boundary_mask: np.ndarray
    boundary_mask: np.ndarray
    cluster_ids: np.ndarray

    @property
    def cluster_count(self) -> int:
        if len(self.cluster_ids) == 0:
            return 0
        return int(np.max(self.cluster_ids)) + 1


def _mad(values: np.ndarray, center: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.median(np.abs(values - center)))


def _local_distance_stats(distances: np.ndarray, local_window: int) -> tuple[np.ndarray, np.ndarray]:
    local_medians = np.zeros(len(distances), dtype=np.float32)
    local_mads = np.zeros(len(distances), dtype=np.float32)
    fallback_median = float(np.median(distances)) if len(distances) else 0.0
    fallback_mad = _mad(distances, fallback_median)

    for order in range(len(distances)):
        start = max(0, order - local_window)
        end = min(len(distances), order + local_window + 1)
        context = np.concatenate([distances[start:order], distances[order + 1 : end]])
        if len(context) == 0:
            local_medians[order] = fallback_median
            local_mads[order] = fallback_mad
            continue
        local_median = float(np.median(context))
        local_medians[order] = local_median
        local_mads[order] = _mad(context, local_median)

    return local_medians, local_mads


def _apply_min_cluster_size(
    boundary_mask: np.ndarray,
    distances: np.ndarray,
    selected_count: int,
    min_cluster_size: int,
) -> np.ndarray:
    if min_cluster_size <= 1 or not np.any(boundary_mask):
        return boundary_mask.astype(bool)

    filtered = boundary_mask.astype(bool).copy()
    while True:
        boundary_orders = np.where(filtered)[0]
        if len(boundary_orders) == 0:
            return filtered

        starts = np.concatenate([[0], boundary_orders + 1])
        ends = np.concatenate([boundary_orders, [selected_count - 1]])
        sizes = ends - starts + 1
        small_positions = np.where(sizes < min_cluster_size)[0]
        if len(small_positions) == 0:
            return filtered

        cluster_position = int(small_positions[0])
        if cluster_position == 0:
            boundary_to_remove = int(boundary_orders[0])
        elif cluster_position == len(sizes) - 1:
            boundary_to_remove = int(boundary_orders[-1])
        else:
            left_boundary = int(boundary_orders[cluster_position - 1])
            right_boundary = int(boundary_orders[cluster_position])
            boundary_to_remove = (
                left_boundary
                if float(distances[left_boundary]) <= float(distances[right_boundary])
                else right_boundary
            )
        filtered[boundary_to_remove] = False


def _candidate_context_stats(
    latents: np.ndarray | None,
    selected_candidate_orders: np.ndarray | None,
    distances: np.ndarray,
    context_window: int,
    mad_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_count = len(distances)
    candidate_gaps = np.ones(edge_count, dtype=np.int32)
    context_counts = np.zeros(edge_count, dtype=np.int32)
    local_medians = np.zeros(edge_count, dtype=np.float32)
    local_mads = np.zeros(edge_count, dtype=np.float32)
    local_scales = np.zeros(edge_count, dtype=np.float32)
    local_thresholds = np.zeros(edge_count, dtype=np.float32)
    expected_distances = np.zeros(edge_count, dtype=np.float32)
    distance_ratios = np.full(edge_count, np.inf, dtype=np.float32)

    if latents is None or selected_candidate_orders is None or edge_count == 0:
        return (
            candidate_gaps,
            context_counts,
            local_medians,
            local_mads,
            local_scales,
            local_thresholds,
            expected_distances,
            distance_ratios,
        )

    selected_orders = selected_candidate_orders.astype(np.int32)
    if len(selected_orders) != edge_count + 1 or len(latents) < 2:
        return (
            candidate_gaps,
            context_counts,
            local_medians,
            local_mads,
            local_scales,
            local_thresholds,
            expected_distances,
            distance_ratios,
        )

    adjacent_distances = np.linalg.norm(np.diff(latents.astype(np.float32), axis=0), axis=1)
    context_window = max(1, int(context_window))
    for edge_order in range(edge_count):
        start = int(min(selected_orders[edge_order], selected_orders[edge_order + 1]))
        end = int(max(selected_orders[edge_order], selected_orders[edge_order + 1]))
        start = max(0, min(start, len(adjacent_distances)))
        end = max(start + 1, min(end, len(adjacent_distances)))
        candidate_gap = max(1, end - start)
        candidate_gaps[edge_order] = candidate_gap

        left = adjacent_distances[max(0, start - context_window) : start]
        right = adjacent_distances[end : min(len(adjacent_distances), end + context_window)]
        context = np.concatenate([left, right])
        if len(context) == 0:
            context = np.concatenate([adjacent_distances[:start], adjacent_distances[end:]])
        if len(context) == 0:
            context = adjacent_distances

        context_counts[edge_order] = int(len(context))
        local_median = float(np.median(context)) if len(context) else 0.0
        local_mad = _mad(context, local_median)
        local_scale = float(1.4826 * local_mad)
        local_threshold = local_median + (float(mad_multiplier) * local_scale)
        expected_distance = local_threshold * candidate_gap
        local_medians[edge_order] = local_median
        local_mads[edge_order] = local_mad
        local_scales[edge_order] = local_scale
        local_thresholds[edge_order] = local_threshold
        expected_distances[edge_order] = expected_distance
        if expected_distance > 1e-8:
            distance_ratios[edge_order] = float(distances[edge_order]) / expected_distance
        elif float(distances[edge_order]) <= 1e-8:
            distance_ratios[edge_order] = 0.0

    return (
        candidate_gaps,
        context_counts,
        local_medians,
        local_mads,
        local_scales,
        local_thresholds,
        expected_distances,
        distance_ratios,
    )


def compute_selected_clusters(
    distances: np.ndarray,
    latents: np.ndarray | None = None,
    selected_candidate_orders: np.ndarray | None = None,
    local_window: int = 4,
    local_mad_multiplier: float = 3.5,
    global_percentile: float = 95.0,
    min_distance_ratio: float = 1.5,
    min_cluster_size: int = 3,
    candidate_context_window: int = 12,
    candidate_mad_multiplier: float = 3.5,
    candidate_distance_ratio: float = 2.0,
) -> SelectedClusterInfo:
    distances = distances.astype(np.float32)
    local_window = max(1, int(local_window))
    min_cluster_size = max(1, int(min_cluster_size))
    candidate_context_window = max(1, int(candidate_context_window))
    selected_count = len(distances) + 1
    if len(distances) == 0:
        return SelectedClusterInfo(
            distance_mean=0.0,
            distance_std=0.0,
            distance_median=0.0,
            distance_mad=0.0,
            threshold=0.0,
            local_window=local_window,
            local_mad_multiplier=float(local_mad_multiplier),
            global_percentile=float(global_percentile),
            min_distance_ratio=float(min_distance_ratio),
            min_cluster_size=min_cluster_size,
            candidate_context_window=candidate_context_window,
            candidate_mad_multiplier=float(candidate_mad_multiplier),
            candidate_distance_ratio=float(candidate_distance_ratio),
            local_medians=np.asarray([], dtype=np.float32),
            local_mads=np.asarray([], dtype=np.float32),
            local_scales=np.asarray([], dtype=np.float32),
            local_thresholds=np.asarray([], dtype=np.float32),
            ratio_thresholds=np.asarray([], dtype=np.float32),
            effective_thresholds=np.asarray([], dtype=np.float32),
            selected_boundary_mask=np.asarray([], dtype=bool),
            candidate_gaps=np.asarray([], dtype=np.int32),
            candidate_context_counts=np.asarray([], dtype=np.int32),
            candidate_local_medians=np.asarray([], dtype=np.float32),
            candidate_local_mads=np.asarray([], dtype=np.float32),
            candidate_local_scales=np.asarray([], dtype=np.float32),
            candidate_local_thresholds=np.asarray([], dtype=np.float32),
            candidate_expected_distances=np.asarray([], dtype=np.float32),
            candidate_distance_ratios=np.asarray([], dtype=np.float32),
            candidate_boundary_mask=np.asarray([], dtype=bool),
            boundary_mask=np.asarray([], dtype=bool),
            cluster_ids=np.zeros(selected_count, dtype=np.int32),
        )

    distance_mean = float(np.mean(distances))
    distance_std = float(np.std(distances))
    distance_median = float(np.median(distances))
    distance_mad = _mad(distances, distance_median)
    global_threshold = float(np.percentile(distances, float(global_percentile)))
    local_medians, local_mads = _local_distance_stats(distances, local_window)
    local_scales = (1.4826 * local_mads).astype(np.float32)
    local_thresholds = local_medians + (float(local_mad_multiplier) * local_scales)
    ratio_thresholds = (local_medians * float(min_distance_ratio)).astype(np.float32)
    effective_thresholds = np.maximum.reduce(
        [
            local_thresholds.astype(np.float32),
            np.full(len(distances), global_threshold, dtype=np.float32),
            ratio_thresholds.astype(np.float32),
        ]
    ).astype(np.float32)
    selected_boundary_mask = (
        (distances >= effective_thresholds)
        & (distances > local_medians)
        & (distance_std > 0)
    )
    (
        candidate_gaps,
        candidate_context_counts,
        candidate_local_medians,
        candidate_local_mads,
        candidate_local_scales,
        candidate_local_thresholds,
        candidate_expected_distances,
        candidate_distance_ratios,
    ) = _candidate_context_stats(
        latents=latents,
        selected_candidate_orders=selected_candidate_orders,
        distances=distances,
        context_window=candidate_context_window,
        mad_multiplier=candidate_mad_multiplier,
    )
    candidate_boundary_mask = selected_boundary_mask & (
        candidate_distance_ratios >= float(candidate_distance_ratio)
    )
    boundary_mask = _apply_min_cluster_size(
        candidate_boundary_mask,
        distances,
        selected_count=selected_count,
        min_cluster_size=min_cluster_size,
    )
    cluster_ids = np.zeros(selected_count, dtype=np.int32)
    current_cluster = 0
    for distance_order, is_boundary in enumerate(boundary_mask):
        if is_boundary:
            current_cluster += 1
        cluster_ids[distance_order + 1] = current_cluster

    return SelectedClusterInfo(
        distance_mean=distance_mean,
        distance_std=distance_std,
        distance_median=distance_median,
        distance_mad=distance_mad,
        threshold=float(global_threshold),
        local_window=local_window,
        local_mad_multiplier=float(local_mad_multiplier),
        global_percentile=float(global_percentile),
        min_distance_ratio=float(min_distance_ratio),
        min_cluster_size=min_cluster_size,
        candidate_context_window=candidate_context_window,
        candidate_mad_multiplier=float(candidate_mad_multiplier),
        candidate_distance_ratio=float(candidate_distance_ratio),
        local_medians=local_medians.astype(np.float32),
        local_mads=local_mads.astype(np.float32),
        local_scales=local_scales.astype(np.float32),
        local_thresholds=local_thresholds.astype(np.float32),
        ratio_thresholds=ratio_thresholds.astype(np.float32),
        effective_thresholds=effective_thresholds.astype(np.float32),
        selected_boundary_mask=selected_boundary_mask.astype(bool),
        candidate_gaps=candidate_gaps.astype(np.int32),
        candidate_context_counts=candidate_context_counts.astype(np.int32),
        candidate_local_medians=candidate_local_medians.astype(np.float32),
        candidate_local_mads=candidate_local_mads.astype(np.float32),
        candidate_local_scales=candidate_local_scales.astype(np.float32),
        candidate_local_thresholds=candidate_local_thresholds.astype(np.float32),
        candidate_expected_distances=candidate_expected_distances.astype(np.float32),
        candidate_distance_ratios=candidate_distance_ratios.astype(np.float32),
        candidate_boundary_mask=candidate_boundary_mask.astype(bool),
        boundary_mask=boundary_mask.astype(bool),
        cluster_ids=cluster_ids,
    )


def cluster_summaries(
    cluster_ids: np.ndarray,
    selected_candidate_orders: np.ndarray,
    selected_frame_indices: np.ndarray,
) -> list[dict[str, int]]:
    summaries: list[dict[str, int]] = []
    if len(cluster_ids) == 0:
        return summaries

    for cluster_id in range(int(np.max(cluster_ids)) + 1):
        positions = np.where(cluster_ids == cluster_id)[0]
        if len(positions) == 0:
            continue
        start_position = int(positions[0])
        end_position = int(positions[-1])
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "selected_count": int(len(positions)),
                "start_selection_order": start_position,
                "end_selection_order": end_position,
                "start_candidate_order": int(selected_candidate_orders[start_position]),
                "end_candidate_order": int(selected_candidate_orders[end_position]),
                "start_frame_index": int(selected_frame_indices[start_position]),
                "end_frame_index": int(selected_frame_indices[end_position]),
            }
        )
    return summaries
