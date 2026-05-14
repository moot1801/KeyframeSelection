from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from keyframe_pipeline.config import SelectionConfig


@dataclass(frozen=True)
class SelectionResult:
    cumulative: np.ndarray
    initial_selected: np.ndarray
    final_selected: np.ndarray
    initial_distances: np.ndarray
    final_distances: np.ndarray
    requested_num_frames: int
    resolved_num_frames: int
    auto_num_frames_summary: dict[str, object] | None = None
    intra_cluster_initial_distances: np.ndarray | None = None
    intra_cluster_final_distances: np.ndarray | None = None
    precluster_boundary_distances: np.ndarray | None = None
    precluster_summary: dict[str, object] | None = None


class SelectionStrategy(ABC):
    name: str

    @abstractmethod
    def select(self, latents: np.ndarray, config: SelectionConfig) -> SelectionResult:
        raise NotImplementedError


def adjacent_l2_distances(latents: np.ndarray) -> np.ndarray:
    if len(latents) < 2:
        return np.asarray([], dtype=np.float32)
    return np.linalg.norm(np.diff(latents, axis=0), axis=1).astype(np.float32)


def cumulative_path_distances(latents: np.ndarray) -> np.ndarray:
    distances = adjacent_l2_distances(latents)
    return np.concatenate([[0.0], np.cumsum(distances)]).astype(np.float32)


def choose_nearest_index(cumulative: np.ndarray, target: float, lower: int, upper: int) -> int:
    if lower > upper:
        raise ValueError("선택 가능한 후보 프레임 범위가 비어 있습니다.")
    nearest = int(np.searchsorted(cumulative, target, side="left"))
    nearest = min(max(nearest, lower), upper)
    scan_lower = max(lower, nearest - 4)
    scan_upper = min(upper, nearest + 4)
    candidates = range(scan_lower, scan_upper + 1)
    return min(candidates, key=lambda index: (abs(float(cumulative[index]) - target), index))


def initial_selection_by_arclength(
    cumulative: np.ndarray,
    num_frames: int,
    include_endpoints: bool,
) -> np.ndarray:
    candidate_count = len(cumulative)
    if num_frames > candidate_count:
        raise ValueError("최종 프레임 수가 선택 대상 전체 프레임 수보다 큽니다.")

    if include_endpoints:
        if num_frames == 2:
            return np.asarray([0, candidate_count - 1], dtype=np.int32)
        targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), num_frames)
        selected = [0]
        for order in range(1, num_frames - 1):
            lower = selected[-1] + 1
            upper = candidate_count - (num_frames - order)
            selected.append(choose_nearest_index(cumulative, float(targets[order]), lower, upper))
        selected.append(candidate_count - 1)
        return np.asarray(selected, dtype=np.int32)

    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), num_frames + 2)[1:-1]
    selected: list[int] = []
    for order, target in enumerate(targets):
        lower = selected[-1] + 1 if selected else 0
        upper = candidate_count - (num_frames - order)
        selected.append(choose_nearest_index(cumulative, float(target), lower, upper))
    return np.asarray(selected, dtype=np.int32)


def selected_distances(latents: np.ndarray, selected_candidate_orders: np.ndarray) -> np.ndarray:
    return adjacent_l2_distances(latents[selected_candidate_orders])


def distance_variance(latents: np.ndarray, selected_candidate_orders: np.ndarray) -> float:
    distances = selected_distances(latents, selected_candidate_orders)
    if len(distances) == 0:
        return 0.0
    return float(np.var(distances))


def distance_cv(distances: np.ndarray) -> float:
    if len(distances) == 0:
        return 0.0
    mean = float(np.mean(distances))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(distances) / mean)


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mad(values: np.ndarray, center: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.median(np.abs(values - center)))


def _precluster_config(config: SelectionConfig) -> dict[str, object]:
    raw_config = config.kwargs.get("precluster")
    if not isinstance(raw_config, dict) or not _as_bool(raw_config.get("enabled"), False):
        return {"enabled": False}

    return {
        "enabled": True,
        "global_percentile": min(100.0, max(0.0, _as_float(raw_config.get("global_percentile"), 99.0))),
        "local_window": max(1, _as_int(raw_config.get("local_window"), 5)),
        "local_mad_multiplier": max(0.0, _as_float(raw_config.get("local_mad_multiplier"), 3.5)),
        "min_local_median_ratio": max(0.0, _as_float(raw_config.get("min_local_median_ratio"), 8.0)),
        "min_local_threshold_ratio": max(0.0, _as_float(raw_config.get("min_local_threshold_ratio"), 3.0)),
        "max_neighbor_high_count": max(0, _as_int(raw_config.get("max_neighbor_high_count"), 1)),
        "min_cluster_frames": max(2, _as_int(raw_config.get("min_cluster_frames"), 30)),
        "min_frames_per_cluster": max(2, _as_int(raw_config.get("min_frames_per_cluster"), 2)),
    }


def _cluster_ranges_from_boundaries(
    candidate_count: int,
    boundary_edges: list[int],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for edge in boundary_edges:
        edge = int(edge)
        ranges.append((start, edge))
        start = edge + 1
    ranges.append((start, candidate_count - 1))
    return [(start, end) for start, end in ranges if start <= end]


def detect_precluster_ranges(
    latents: np.ndarray,
    config: SelectionConfig,
) -> dict[str, object]:
    precluster_config = _precluster_config(config)
    candidate_count = len(latents)
    fallback_ranges = [(0, candidate_count - 1)] if candidate_count else []
    if not _as_bool(precluster_config.get("enabled"), False) or candidate_count < 2:
        return {
            "enabled": False,
            "config": precluster_config,
            "boundary_edges": [],
            "ranges": fallback_ranges,
            "cluster_count": len(fallback_ranges),
            "candidates": [],
        }

    distances = adjacent_l2_distances(latents)
    global_threshold = float(np.percentile(distances, float(precluster_config["global_percentile"])))
    local_window = int(precluster_config["local_window"])
    local_mad_multiplier = float(precluster_config["local_mad_multiplier"])
    min_local_median_ratio = float(precluster_config["min_local_median_ratio"])
    min_local_threshold_ratio = float(precluster_config["min_local_threshold_ratio"])
    max_neighbor_high_count = int(precluster_config["max_neighbor_high_count"])
    min_cluster_frames = int(precluster_config["min_cluster_frames"])
    candidate_rows: list[dict[str, float | int | bool]] = []
    boundary_edges: list[int] = []
    current_start = 0

    for edge, distance in enumerate(distances):
        left = distances[max(0, edge - local_window) : edge]
        right = distances[edge + 1 : min(len(distances), edge + local_window + 1)]
        context = np.concatenate([left, right])
        local_median = float(np.median(context)) if len(context) else 0.0
        local_mad = _mad(context, local_median)
        local_scale = float(1.4826 * local_mad)
        local_threshold = local_median + (local_mad_multiplier * local_scale)
        ratio_to_local_median = float(distance) / max(local_median, 1e-12)
        ratio_to_local_threshold = float(distance) / max(local_threshold, 1e-12)
        neighbor_high_count = int(np.sum(context >= global_threshold)) if len(context) else 0
        is_candidate = bool(
            float(distance) >= global_threshold
            and ratio_to_local_median >= min_local_median_ratio
            and ratio_to_local_threshold >= min_local_threshold_ratio
            and neighbor_high_count <= max_neighbor_high_count
        )
        left_count = edge - current_start + 1
        right_count = candidate_count - (edge + 1)
        accepted = bool(is_candidate and left_count >= min_cluster_frames and right_count >= min_cluster_frames)
        if is_candidate:
            candidate_rows.append(
                {
                    "edge": int(edge),
                    "distance": float(distance),
                    "local_median": local_median,
                    "local_threshold": float(local_threshold),
                    "ratio_to_local_median": ratio_to_local_median,
                    "ratio_to_local_threshold": ratio_to_local_threshold,
                    "neighbor_high_count": neighbor_high_count,
                    "left_count": int(left_count),
                    "right_count": int(right_count),
                    "accepted": accepted,
                }
            )
        if accepted:
            boundary_edges.append(int(edge))
            current_start = edge + 1

    ranges = _cluster_ranges_from_boundaries(candidate_count, boundary_edges)
    print(
        "  - precluster: "
        f"enabled=true, boundary_count={len(boundary_edges)}, "
        f"cluster_count={len(ranges)}, global_threshold={global_threshold:.6f}"
    )
    for row in candidate_rows:
        print(
            "  - precluster candidate: "
            f"edge={row['edge']}, distance={float(row['distance']):.6f}, "
            f"local_ratio={float(row['ratio_to_local_median']):.6f}, "
            f"threshold_ratio={float(row['ratio_to_local_threshold']):.6f}, "
            f"neighbor_high_count={row['neighbor_high_count']}, accepted={row['accepted']}"
        )

    return {
        "enabled": True,
        "config": precluster_config,
        "global_threshold": global_threshold,
        "boundary_edges": boundary_edges,
        "ranges": ranges,
        "cluster_count": len(ranges),
        "candidates": candidate_rows,
    }


def _precluster_is_active(precluster_summary: dict[str, object] | None) -> bool:
    if not precluster_summary or not _as_bool(precluster_summary.get("enabled"), False):
        return False
    return int(precluster_summary.get("cluster_count", 1)) > 1


def _range_arclength(latents: np.ndarray, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    return float(cumulative_path_distances(latents[start : end + 1])[-1])


def allocate_frames_to_ranges(
    latents: np.ndarray,
    ranges: list[tuple[int, int]],
    num_frames: int,
    min_frames_per_cluster: int,
) -> list[int]:
    capacities = [end - start + 1 for start, end in ranges]
    min_counts = [min(max(1, min_frames_per_cluster), capacity) for capacity in capacities]
    if num_frames < sum(min_counts):
        num_frames = sum(min_counts)

    weights = np.asarray([_range_arclength(latents, start, end) for start, end in ranges], dtype=np.float64)
    if float(np.sum(weights)) <= 1e-12:
        weights = np.asarray(capacities, dtype=np.float64)
    weights = weights / max(float(np.sum(weights)), 1e-12)

    ideal = weights * float(num_frames)
    counts = [min(capacity, max(min_count, int(np.floor(value)))) for capacity, min_count, value in zip(capacities, min_counts, ideal)]
    while sum(counts) < num_frames:
        candidates = [index for index, count in enumerate(counts) if count < capacities[index]]
        if not candidates:
            break
        best = max(candidates, key=lambda index: (ideal[index] - counts[index], weights[index]))
        counts[best] += 1
    while sum(counts) > num_frames:
        candidates = [index for index, count in enumerate(counts) if count > min_counts[index]]
        if not candidates:
            break
        best = max(candidates, key=lambda index: (counts[index] - ideal[index], -weights[index]))
        counts[best] -= 1
    return counts


def cluster_ids_for_selected(
    selected_candidate_orders: np.ndarray,
    ranges: list[tuple[int, int]],
) -> np.ndarray:
    cluster_ids = np.zeros(len(selected_candidate_orders), dtype=np.int32)
    for order, candidate_order in enumerate(selected_candidate_orders):
        for cluster_id, (start, end) in enumerate(ranges):
            if start <= int(candidate_order) <= end:
                cluster_ids[order] = cluster_id
                break
    return cluster_ids


def split_selected_distances_by_cluster(
    latents: np.ndarray,
    selected_candidate_orders: np.ndarray,
    ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if len(selected_candidate_orders) < 2:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    distances = selected_distances(latents, selected_candidate_orders)
    cluster_ids = cluster_ids_for_selected(selected_candidate_orders, ranges)
    intra_mask = cluster_ids[:-1] == cluster_ids[1:]
    return distances[intra_mask], distances[~intra_mask]


def _minimum_auto_num_frames(
    candidate_count: int,
    precluster_active: bool,
    precluster_ranges: list[tuple[int, int]],
    min_frames_per_cluster: int,
) -> int:
    if candidate_count <= 0:
        return 0
    if not precluster_active:
        return min(candidate_count, 2)
    capacities = [end - start + 1 for start, end in precluster_ranges]
    minimum = sum(min(max(1, min_frames_per_cluster), capacity) for capacity in capacities)
    return min(candidate_count, max(1, minimum))


def _auto_num_frame_bounds(
    raw_auto_config: dict[str, object],
    candidate_count: int,
    minimum_frames: int,
) -> tuple[bool, int, int]:
    frame_limit_enabled = _as_bool(raw_auto_config.get("frame_limit_enabled"), False)
    if frame_limit_enabled:
        min_frames = max(minimum_frames, _as_int(raw_auto_config.get("min_frames"), minimum_frames))
        max_frames = max(min_frames, _as_int(raw_auto_config.get("max_frames"), candidate_count))
        max_frames = min(max_frames, candidate_count)
        min_frames = min(min_frames, max_frames)
        return True, min_frames, max_frames
    auto_max_frames = min(max(candidate_count // 5, 5), 25)
    max_frames = min(candidate_count, max(minimum_frames, auto_max_frames))
    return False, minimum_frames, max_frames


def _range_adjacent_distances(latents: np.ndarray, start: int, end: int) -> np.ndarray:
    if end <= start:
        return np.asarray([], dtype=np.float32)
    return adjacent_l2_distances(latents[start : end + 1])


def _range_base_step(latents: np.ndarray, start: int, end: int) -> float:
    distances = _range_adjacent_distances(latents, start, end)
    if len(distances) == 0:
        return 0.0
    median = float(np.median(distances))
    if median > 1e-12:
        return median
    mean = float(np.mean(distances))
    return mean if mean > 1e-12 else 0.0


def _target_density_num_frames(
    latents: np.ndarray,
    ranges: list[tuple[int, int]],
    target_gap_ratio: float,
    min_frames_per_cluster: int,
    min_frames: int,
    max_frames: int,
) -> tuple[int, list[dict[str, float | int]]]:
    per_range: list[dict[str, float | int]] = []
    total = 0
    for cluster_id, (start, end) in enumerate(ranges):
        capacity = end - start + 1
        minimum = min(max(1, min_frames_per_cluster), capacity)
        distances = _range_adjacent_distances(latents, start, end)
        base_step = _range_base_step(latents, start, end)
        arc_length = float(np.sum(distances)) if len(distances) else 0.0
        if base_step <= 1e-12 or target_gap_ratio <= 1e-12:
            count = minimum
        else:
            count = int(round(arc_length / (base_step * target_gap_ratio))) + 1
            count = max(minimum, count)
        count = min(capacity, count)
        total += count
        per_range.append(
            {
                "cluster_id": int(cluster_id),
                "start_candidate_order": int(start),
                "end_candidate_order": int(end),
                "candidate_count": int(capacity),
                "base_step": float(base_step),
                "arc_length": float(arc_length),
                "target_num_frames": int(count),
            }
        )
    return min(max(total, min_frames), max_frames), per_range


def _selected_gap_ratio_summary(
    latents: np.ndarray,
    selected_candidate_orders: np.ndarray,
    ranges: list[tuple[int, int]],
) -> tuple[float, list[dict[str, float | int]]]:
    weighted_sum = 0.0
    weight_sum = 0
    rows: list[dict[str, float | int]] = []
    for cluster_id, (start, end) in enumerate(ranges):
        cluster_selected = selected_candidate_orders[
            (selected_candidate_orders >= start) & (selected_candidate_orders <= end)
        ]
        distances = selected_distances(latents, cluster_selected)
        base_step = _range_base_step(latents, start, end)
        mean_distance = float(np.mean(distances)) if len(distances) else 0.0
        gap_ratio = mean_distance / base_step if base_step > 1e-12 else 0.0
        if len(distances):
            weighted_sum += gap_ratio * len(distances)
            weight_sum += len(distances)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "selected_count": int(len(cluster_selected)),
                "distance_count": int(len(distances)),
                "base_step": float(base_step),
                "mean_selected_distance": float(mean_distance),
                "gap_ratio": float(gap_ratio),
            }
        )
    return (weighted_sum / weight_sum if weight_sum else 0.0), rows


def select_frames_with_preclusters(
    latents: np.ndarray,
    ranges: list[tuple[int, int]],
    num_frames: int,
    include_endpoints: bool,
    iterations: int,
    window: int,
    refine: bool,
    min_frames_per_cluster: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    counts = allocate_frames_to_ranges(
        latents=latents,
        ranges=ranges,
        num_frames=num_frames,
        min_frames_per_cluster=min_frames_per_cluster,
    )
    initial_parts: list[np.ndarray] = []
    final_parts: list[np.ndarray] = []
    allocation: list[dict[str, int]] = []
    for cluster_id, ((start, end), count) in enumerate(zip(ranges, counts)):
        local_latents = latents[start : end + 1]
        local_cumulative = cumulative_path_distances(local_latents)
        local_initial = initial_selection_by_arclength(
            cumulative=local_cumulative,
            num_frames=count,
            include_endpoints=include_endpoints,
        )
        local_final = local_initial
        if refine:
            local_final = refine_selection_by_local_search(
                latents=local_latents,
                selected_candidate_orders=local_initial,
                iterations=iterations,
                window=window,
                include_endpoints=include_endpoints,
            )
        initial_parts.append(local_initial + start)
        final_parts.append(local_final + start)
        allocation.append(
            {
                "cluster_id": int(cluster_id),
                "start_candidate_order": int(start),
                "end_candidate_order": int(end),
                "candidate_count": int(end - start + 1),
                "num_frames": int(count),
            }
        )
    initial_selected = np.concatenate(initial_parts).astype(np.int32)
    final_selected = np.concatenate(final_parts).astype(np.int32)
    return initial_selected, final_selected, allocation


def resolve_num_frames_by_density_score(
    latents: np.ndarray,
    config: SelectionConfig,
    cumulative: np.ndarray,
    precluster_summary: dict[str, object] | None = None,
) -> tuple[int, dict[str, object] | None]:
    raw_auto_config = config.kwargs.get("auto_num_frames")
    if not isinstance(raw_auto_config, dict) or not _as_bool(raw_auto_config.get("enabled"), False):
        return config.num_frames, None

    candidate_count = len(latents)
    search_step = max(1, _as_int(raw_auto_config.get("search_step"), 10))
    refine_candidates = _as_bool(raw_auto_config.get("refine_candidates"), False)
    precluster_active = _precluster_is_active(precluster_summary)
    precluster_ranges = (
        [(int(start), int(end)) for start, end in precluster_summary["ranges"]]
        if precluster_active and precluster_summary is not None
        else []
    )
    min_frames_per_cluster = int(
        (precluster_summary or {}).get("config", {}).get("min_frames_per_cluster", 2)
        if precluster_summary
        else 2
    )
    selection_ranges = precluster_ranges if precluster_active else [(0, candidate_count - 1)]
    minimum_frames = _minimum_auto_num_frames(
        candidate_count=candidate_count,
        precluster_active=precluster_active,
        precluster_ranges=precluster_ranges,
        min_frames_per_cluster=min_frames_per_cluster,
    )
    frame_limit_enabled, min_frames, max_frames = _auto_num_frame_bounds(
        raw_auto_config=raw_auto_config,
        candidate_count=candidate_count,
        minimum_frames=minimum_frames,
    )
    target_gap_ratio = max(0.0, _as_float(raw_auto_config.get("target_gap_ratio"), 10.0))
    soft_cv_target = max(0.0, _as_float(raw_auto_config.get("soft_cv_target"), 0.25))
    density_weight = max(0.0, _as_float(raw_auto_config.get("density_weight"), 1.0))
    cv_weight = max(0.0, _as_float(raw_auto_config.get("cv_weight"), 0.5))
    target_num_frames, target_density_ranges = _target_density_num_frames(
        latents=latents,
        ranges=selection_ranges,
        target_gap_ratio=target_gap_ratio,
        min_frames_per_cluster=min_frames_per_cluster if precluster_active else 2,
        min_frames=min_frames,
        max_frames=max_frames,
    )

    candidate_values = list(range(min_frames, max_frames + 1, search_step))
    if target_num_frames is not None and target_num_frames not in candidate_values:
        candidate_values.append(target_num_frames)
    if candidate_values[-1] != max_frames:
        candidate_values.append(max_frames)
    candidate_values = sorted(set(candidate_values))

    best_num_frames: int | None = None
    best_cv: float | None = None
    best_density_error: float | None = None
    best_score: float | None = None
    evaluated: list[dict[str, object]] = []
    print(
        "  - auto num_frames enabled: "
        f"method=target_density_score, frame_limit_enabled={frame_limit_enabled}, "
        f"min_frames={min_frames}, max_frames={max_frames}, "
        f"search_step={search_step}, refine_candidates={refine_candidates}, "
        f"precluster_active={precluster_active}"
    )
    print(
        "  - auto num_frames density target: "
        f"target_gap_ratio={target_gap_ratio:.6f}, target_num_frames={target_num_frames}"
    )
    print(
        "  - auto num_frames score weights: "
        f"density_weight={density_weight:.6f}, cv_weight={cv_weight:.6f}, "
        f"soft_cv_target={soft_cv_target:.6f}"
    )
    for num_frames in candidate_values:
        if precluster_active:
            _initial_selected, selected, allocation = select_frames_with_preclusters(
                latents=latents,
                ranges=precluster_ranges,
                num_frames=num_frames,
                include_endpoints=config.include_endpoints,
                iterations=config.local_refine_iterations,
                window=config.local_refine_window,
                refine=refine_candidates,
                min_frames_per_cluster=min_frames_per_cluster,
            )
            distances, boundary_distances = split_selected_distances_by_cluster(
                latents=latents,
                selected_candidate_orders=selected,
                ranges=precluster_ranges,
            )
        else:
            initial_selected = initial_selection_by_arclength(
                cumulative=cumulative,
                num_frames=num_frames,
                include_endpoints=config.include_endpoints,
            )
            selected = initial_selected
            allocation = []
            boundary_distances = np.asarray([], dtype=np.float32)
            if refine_candidates:
                selected = refine_selection_by_local_search(
                    latents=latents,
                    selected_candidate_orders=initial_selected,
                    iterations=config.local_refine_iterations,
                    window=config.local_refine_window,
                    include_endpoints=config.include_endpoints,
                )
            distances = selected_distances(latents, selected)
        gap_ratio, gap_ratio_by_cluster = _selected_gap_ratio_summary(
            latents=latents,
            selected_candidate_orders=selected,
            ranges=selection_ranges,
        )
        cv = distance_cv(distances)
        density_error = abs(float(gap_ratio) - target_gap_ratio)
        density_error_norm = float(density_error) / max(target_gap_ratio, 1e-12)
        cv_penalty = max(0.0, cv - soft_cv_target)
        score = (density_weight * density_error_norm) + (cv_weight * cv_penalty)
        evaluated.append(
            {
                "num_frames": int(num_frames),
                "cv": float(cv),
                "mean": float(np.mean(distances)) if len(distances) else 0.0,
                "variance": float(np.var(distances)) if len(distances) else 0.0,
                "boundary_distance_count": int(len(boundary_distances)),
                "gap_ratio": float(gap_ratio),
                "density_error": float(density_error),
                "density_error_norm": float(density_error_norm),
                "cv_penalty": float(cv_penalty),
                "score": float(score),
                "gap_ratio_by_cluster": gap_ratio_by_cluster,
            }
        )
        print(
            "  - auto num_frames candidate: "
            f"num_frames={num_frames}, cv={cv:.6f}, "
            f"gap_ratio={gap_ratio:.6f}, score={score:.6f}"
        )
        if (
            best_score is None
            or score < best_score - 1e-12
            or (abs(score - best_score) <= 1e-12 and density_error < float(best_density_error or np.inf))
        ):
            best_num_frames = int(num_frames)
            best_cv = cv
            best_density_error = float(density_error)
            best_score = float(score)

    if best_num_frames is None:
        best_row = min(evaluated, key=lambda row: (float(row["score"]), float(row["density_error_norm"])))
        best_num_frames = int(best_row["num_frames"])
        best_cv = float(best_row["cv"])
        best_density_error = float(best_row["density_error"])
        best_score = float(best_row["score"])
        fallback_reason = "no_candidate_score_available"
    else:
        fallback_reason = None

    summary: dict[str, object] = {
        "enabled": True,
        "method": "target_density_score",
        "frame_limit_enabled": bool(frame_limit_enabled),
        "min_frames": int(min_frames),
        "max_frames": int(max_frames),
        "search_step": int(search_step),
        "refine_candidates": bool(refine_candidates),
        "precluster_active": bool(precluster_active),
        "soft_cv_target": float(soft_cv_target),
        "density_weight": float(density_weight),
        "cv_weight": float(cv_weight),
        "target_gap_ratio": float(target_gap_ratio),
        "target_num_frames": int(target_num_frames) if target_num_frames is not None else None,
        "target_density_ranges": target_density_ranges,
        "selected_num_frames": int(best_num_frames),
        "selected_cv": float(best_cv) if best_cv is not None else None,
        "selected_density_error": best_density_error,
        "selected_score": best_score,
        "fallback_reason": fallback_reason,
        "evaluated": evaluated,
    }
    print(
        "  - auto num_frames selected: "
        f"num_frames={best_num_frames}, cv={best_cv:.6f}, fallback_reason={fallback_reason}"
    )
    return best_num_frames, summary


def refine_selection_by_local_search(
    latents: np.ndarray,
    selected_candidate_orders: np.ndarray,
    iterations: int,
    window: int,
    include_endpoints: bool,
) -> np.ndarray:
    if iterations == 0 or len(selected_candidate_orders) <= 2:
        print(
            "  - local refinement skipped: "
            f"iterations={iterations}, selected_count={len(selected_candidate_orders)}"
        )
        return selected_candidate_orders.copy()

    selected = selected_candidate_orders.astype(np.int32).copy()
    fixed_positions = {0, len(selected) - 1} if include_endpoints else set()
    current_objective = distance_variance(latents, selected)
    print(
        "  - local refinement setup: "
        f"iterations={iterations}, window={window}, selected_count={len(selected)}, "
        f"fixed_positions={sorted(fixed_positions)}, initial_variance={current_objective:.6f}"
    )

    for iteration in range(iterations):
        changed = False
        improvement_count = 0
        checked_candidate_count = 0
        iteration_start_objective = current_objective
        for position in range(len(selected)):
            if position in fixed_positions:
                continue

            original_index = int(selected[position])
            lower = int(selected[position - 1] + 1) if position > 0 else 0
            upper = int(selected[position + 1] - 1) if position < len(selected) - 1 else len(latents) - 1
            if lower > upper:
                print(
                    "  - local refinement position skipped: "
                    f"iteration={iteration + 1}, position={position}, reason=empty_range"
                )
                continue

            if window > 0:
                center = int(selected[position])
                lower = max(lower, center - window)
                upper = min(upper, center + window)
            candidate_count = max(0, upper - lower)
            checked_candidate_count += candidate_count

            best_index = int(selected[position])
            best_objective = current_objective
            for candidate_order in range(lower, upper + 1):
                if candidate_order == selected[position]:
                    continue
                trial = selected.copy()
                trial[position] = candidate_order
                objective = distance_variance(latents, trial)
                if objective + 1e-12 < best_objective:
                    best_objective = objective
                    best_index = candidate_order

            if best_index != selected[position]:
                selected[position] = best_index
                current_objective = best_objective
                changed = True
                improvement_count += 1
                print(
                    "  - local refinement improved: "
                    f"iteration={iteration + 1}, position={position}, "
                    f"candidate_order={original_index}->{best_index}, "
                    f"variance={current_objective:.6f}, search_range={lower}-{upper}"
                )

        print(
            "  - local refinement iteration summary: "
            f"iteration={iteration + 1}/{iterations}, checked_candidates={checked_candidate_count}, "
            f"improvements={improvement_count}, "
            f"variance={iteration_start_objective:.6f}->{current_objective:.6f}"
        )

        if not changed:
            print(f"  - local refinement early stop: iteration={iteration + 1}, no_improvement=true")
            break

    return selected


class ArclengthLocalRefineSelectionStrategy(SelectionStrategy):
    name = "arclength_local_refine"

    def select(self, latents: np.ndarray, config: SelectionConfig) -> SelectionResult:
        print(
            "  - selection setup: "
            f"latents_shape={latents.shape}, num_frames={config.num_frames}, "
            f"distance_metric={config.distance_metric}, include_endpoints={config.include_endpoints}"
        )
        cumulative = cumulative_path_distances(latents)
        print(f"  - cumulative latent path length: {float(cumulative[-1]):.6f}")
        precluster_summary = detect_precluster_ranges(latents=latents, config=config)
        precluster_active = _precluster_is_active(precluster_summary)
        resolved_num_frames, auto_num_frames_summary = resolve_num_frames_by_density_score(
            latents=latents,
            config=config,
            cumulative=cumulative,
            precluster_summary=precluster_summary,
        )
        if precluster_active:
            precluster_ranges = [(int(start), int(end)) for start, end in precluster_summary["ranges"]]
            min_frames_per_cluster = int(precluster_summary["config"].get("min_frames_per_cluster", 2))
            initial_selected, final_selected, allocation = select_frames_with_preclusters(
                latents=latents,
                ranges=precluster_ranges,
                num_frames=resolved_num_frames,
                include_endpoints=config.include_endpoints,
                iterations=config.local_refine_iterations,
                window=config.local_refine_window,
                refine=True,
                min_frames_per_cluster=min_frames_per_cluster,
            )
            precluster_summary = {**precluster_summary, "allocation": allocation}
        else:
            initial_selected = initial_selection_by_arclength(
                cumulative=cumulative,
                num_frames=resolved_num_frames,
                include_endpoints=config.include_endpoints,
            )
            final_selected = refine_selection_by_local_search(
                latents=latents,
                selected_candidate_orders=initial_selected,
                iterations=config.local_refine_iterations,
                window=config.local_refine_window,
                include_endpoints=config.include_endpoints,
            )
        initial_distances = selected_distances(latents, initial_selected)
        print(
            "  - initial selection: "
            f"selected_count={len(initial_selected)}, "
            f"first_candidate_order={int(initial_selected[0])}, "
            f"last_candidate_order={int(initial_selected[-1])}, "
            f"distance_mean={float(np.mean(initial_distances)) if len(initial_distances) else 0.0:.6f}, "
            f"distance_variance={float(np.var(initial_distances)) if len(initial_distances) else 0.0:.6f}"
        )
        final_distances = selected_distances(latents, final_selected)
        if precluster_active:
            intra_cluster_initial_distances, _initial_boundary_distances = split_selected_distances_by_cluster(
                latents=latents,
                selected_candidate_orders=initial_selected,
                ranges=precluster_ranges,
            )
            intra_cluster_final_distances, precluster_boundary_distances = split_selected_distances_by_cluster(
                latents=latents,
                selected_candidate_orders=final_selected,
                ranges=precluster_ranges,
            )
        else:
            intra_cluster_initial_distances = initial_distances
            intra_cluster_final_distances = final_distances
            precluster_boundary_distances = np.asarray([], dtype=np.float32)
        print(
            "  - final selection: "
            f"selected_count={len(final_selected)}, "
            f"first_candidate_order={int(final_selected[0])}, "
            f"last_candidate_order={int(final_selected[-1])}, "
            f"distance_mean={float(np.mean(final_distances)) if len(final_distances) else 0.0:.6f}, "
            f"distance_variance={float(np.var(final_distances)) if len(final_distances) else 0.0:.6f}, "
            f"intra_cluster_cv={distance_cv(intra_cluster_final_distances):.6f}"
        )
        return SelectionResult(
            cumulative=cumulative,
            initial_selected=initial_selected,
            final_selected=final_selected,
            initial_distances=initial_distances,
            final_distances=final_distances,
            requested_num_frames=config.num_frames,
            resolved_num_frames=resolved_num_frames,
            auto_num_frames_summary=auto_num_frames_summary,
            intra_cluster_initial_distances=intra_cluster_initial_distances,
            intra_cluster_final_distances=intra_cluster_final_distances,
            precluster_boundary_distances=precluster_boundary_distances,
            precluster_summary=precluster_summary,
        )


SELECTION_STRATEGIES: dict[str, SelectionStrategy] = {
    strategy.name: strategy
    for strategy in (
        ArclengthLocalRefineSelectionStrategy(),
    )
}


def build_selector(config: SelectionConfig) -> SelectionStrategy:
    try:
        return SELECTION_STRATEGIES[config.name]
    except KeyError as exc:
        available = ", ".join(sorted(SELECTION_STRATEGIES))
        raise ValueError(f"지원하지 않는 selection 전략입니다: {config.name}. available={available}") from exc
