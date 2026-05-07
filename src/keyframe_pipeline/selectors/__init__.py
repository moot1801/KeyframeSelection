from __future__ import annotations

import numpy as np

from keyframe_pipeline.config import SelectionConfig
from keyframe_pipeline.loading import instantiate_class, require_method


def build_selector(config: SelectionConfig) -> object:
    strategy = instantiate_class(module_path=config.module, class_name=config.class_name)
    require_method(strategy, "select", f"{config.module}.{config.class_name}")
    return strategy


def validate_selection_result(result: object) -> object:
    for name in ("cumulative", "initial_selected", "final_selected", "initial_distances", "final_distances"):
        if not hasattr(result, name):
            raise TypeError(f"selector.select() 결과에는 '{name}' 속성이 필요합니다.")
        value = getattr(result, name)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"selector.select() 결과의 '{name}'은 np.ndarray여야 합니다.")
    return result
