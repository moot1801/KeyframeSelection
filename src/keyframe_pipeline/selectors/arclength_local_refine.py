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


def refine_selection_by_local_search(
    latents: np.ndarray,
    selected_candidate_orders: np.ndarray,
    iterations: int,
    window: int,
    include_endpoints: bool,
) -> np.ndarray:
    if iterations == 0 or len(selected_candidate_orders) <= 2:
        return selected_candidate_orders.copy()

    selected = selected_candidate_orders.astype(np.int32).copy()
    fixed_positions = {0, len(selected) - 1} if include_endpoints else set()
    current_objective = distance_variance(latents, selected)

    for _ in range(iterations):
        changed = False
        for position in range(len(selected)):
            if position in fixed_positions:
                continue

            lower = int(selected[position - 1] + 1) if position > 0 else 0
            upper = int(selected[position + 1] - 1) if position < len(selected) - 1 else len(latents) - 1
            if lower > upper:
                continue

            if window > 0:
                center = int(selected[position])
                lower = max(lower, center - window)
                upper = min(upper, center + window)

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

        if not changed:
            break

    return selected


class ArclengthLocalRefineSelectionStrategy(SelectionStrategy):
    name = "arclength_local_refine"

    def select(self, latents: np.ndarray, config: SelectionConfig) -> SelectionResult:
        cumulative = cumulative_path_distances(latents)
        initial_selected = initial_selection_by_arclength(
            cumulative=cumulative,
            num_frames=config.num_frames,
            include_endpoints=config.include_endpoints,
        )
        initial_distances = selected_distances(latents, initial_selected)
        final_selected = refine_selection_by_local_search(
            latents=latents,
            selected_candidate_orders=initial_selected,
            iterations=config.local_refine_iterations,
            window=config.local_refine_window,
            include_endpoints=config.include_endpoints,
        )
        final_distances = selected_distances(latents, final_selected)
        return SelectionResult(
            cumulative=cumulative,
            initial_selected=initial_selected,
            final_selected=final_selected,
            initial_distances=initial_distances,
            final_distances=final_distances,
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
