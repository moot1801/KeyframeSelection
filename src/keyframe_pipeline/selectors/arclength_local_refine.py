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


def resolve_num_frames_by_cv(
    latents: np.ndarray,
    config: SelectionConfig,
    cumulative: np.ndarray,
) -> tuple[int, dict[str, object] | None]:
    raw_auto_config = config.kwargs.get("auto_num_frames")
    if not isinstance(raw_auto_config, dict) or not _as_bool(raw_auto_config.get("enabled"), False):
        return config.num_frames, None

    candidate_count = len(latents)
    min_frames = max(2, _as_int(raw_auto_config.get("min_frames"), config.num_frames))
    max_frames = max(min_frames, _as_int(raw_auto_config.get("max_frames"), candidate_count))
    max_frames = min(max_frames, candidate_count)
    search_step = max(1, _as_int(raw_auto_config.get("search_step"), 10))
    cv_threshold = max(0.0, _as_float(raw_auto_config.get("cv_threshold"), 0.2))
    refine_candidates = _as_bool(raw_auto_config.get("refine_candidates"), False)

    candidate_values = list(range(min_frames, max_frames + 1, search_step))
    if candidate_values[-1] != max_frames:
        candidate_values.append(max_frames)

    best_num_frames: int | None = None
    best_cv: float | None = None
    evaluated: list[dict[str, float | int | bool]] = []
    print(
        "  - auto num_frames enabled: "
        f"method=max_k_under_cv, min_frames={min_frames}, max_frames={max_frames}, "
        f"cv_threshold={cv_threshold:.6f}, search_step={search_step}, "
        f"refine_candidates={refine_candidates}"
    )
    for num_frames in candidate_values:
        initial_selected = initial_selection_by_arclength(
            cumulative=cumulative,
            num_frames=num_frames,
            include_endpoints=config.include_endpoints,
        )
        selected = initial_selected
        if refine_candidates:
            selected = refine_selection_by_local_search(
                latents=latents,
                selected_candidate_orders=initial_selected,
                iterations=config.local_refine_iterations,
                window=config.local_refine_window,
                include_endpoints=config.include_endpoints,
            )
        distances = selected_distances(latents, selected)
        cv = distance_cv(distances)
        passed = cv <= cv_threshold
        evaluated.append(
            {
                "num_frames": int(num_frames),
                "cv": float(cv),
                "mean": float(np.mean(distances)) if len(distances) else 0.0,
                "variance": float(np.var(distances)) if len(distances) else 0.0,
                "passed": bool(passed),
            }
        )
        print(
            "  - auto num_frames candidate: "
            f"num_frames={num_frames}, cv={cv:.6f}, "
            f"threshold={cv_threshold:.6f}, passed={passed}"
        )
        if passed:
            best_num_frames = int(num_frames)
            best_cv = cv

    if best_num_frames is None:
        best_row = min(evaluated, key=lambda row: (float(row["cv"]), -int(row["num_frames"])))
        best_num_frames = int(best_row["num_frames"])
        best_cv = float(best_row["cv"])
        fallback_reason = "no_candidate_under_cv_threshold"
    else:
        fallback_reason = None

    summary: dict[str, object] = {
        "enabled": True,
        "method": "max_k_under_cv",
        "cv_threshold": float(cv_threshold),
        "min_frames": int(min_frames),
        "max_frames": int(max_frames),
        "search_step": int(search_step),
        "refine_candidates": bool(refine_candidates),
        "selected_num_frames": int(best_num_frames),
        "selected_cv": float(best_cv) if best_cv is not None else None,
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
        resolved_num_frames, auto_num_frames_summary = resolve_num_frames_by_cv(
            latents=latents,
            config=config,
            cumulative=cumulative,
        )
        initial_selected = initial_selection_by_arclength(
            cumulative=cumulative,
            num_frames=resolved_num_frames,
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
        final_selected = refine_selection_by_local_search(
            latents=latents,
            selected_candidate_orders=initial_selected,
            iterations=config.local_refine_iterations,
            window=config.local_refine_window,
            include_endpoints=config.include_endpoints,
        )
        final_distances = selected_distances(latents, final_selected)
        print(
            "  - final selection: "
            f"selected_count={len(final_selected)}, "
            f"first_candidate_order={int(final_selected[0])}, "
            f"last_candidate_order={int(final_selected[-1])}, "
            f"distance_mean={float(np.mean(final_distances)) if len(final_distances) else 0.0:.6f}, "
            f"distance_variance={float(np.var(final_distances)) if len(final_distances) else 0.0:.6f}"
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
