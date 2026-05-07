from __future__ import annotations

from keyframe_pipeline.config import VisualizationConfig
from keyframe_pipeline.loading import instantiate_class, require_method
from keyframe_pipeline.visualizers.plotly_latent import save_frame_index_comparison_plot


def build_visualizer(config: VisualizationConfig) -> object:
    strategy = instantiate_class(module_path=config.module, class_name=config.class_name)
    require_method(strategy, "save", f"{config.module}.{config.class_name}")
    return strategy

