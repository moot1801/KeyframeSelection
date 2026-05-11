# Repository Guidelines

## Project Structure & Module Organization

- `src/keyframe_selection.py` is the CLI entry point for the keyframe selection pipeline.
- `src/keyframe_pipeline/` contains the main package: config loading, video processing, training, output writing, selectors, visualizers, losses, optimizers, and models.
- Top-level utilities such as `src/video_split.py`, `src/latent_distance_report.py`, and `src/latent_distance_insights.py` support dataset creation and latent analysis.
- `configs/*.yaml` defines dataset, model, training, selection, output, and visualization settings.
- `scripts/` contains runnable shell wrappers. `data/` holds input videos or extracted frames; `artifacts/` stores generated checkpoints, latents, plots, and keyframe outputs.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install Python dependencies in the active environment.
- `bash scripts/run_keyframe_selection.sh --config configs/drone_gray.yaml`: run the full pipeline with one YAML config via the repository `KS/bin/python` wrapper.
- `bash scripts/run.sh`: run the standard config batch listed in the script.
- `python src/video_split.py`: run the frame extraction utility when preparing frame datasets.

## Coding Style & Naming Conventions

- Use Python 3 style with 4-space indentation, clear type-oriented names, and small functions that match the current module boundaries.
- Keep strategy implementations replaceable through YAML `module` and `class_name` fields.
- Use snake_case for modules, functions, variables, config keys, and output files; use PascalCase for classes such as `ConvAutoEncoder`.
- Prefer minimal diffs and follow nearby patterns before adding new abstractions.

## Testing Guidelines

- No dedicated test suite is currently present. Validate changes by running the smallest relevant pipeline config and checking generated CSV, JSON, HTML, image, and MP4 outputs under `artifacts/`.
- For new test coverage, add focused `pytest` tests under `tests/` with names like `test_config.py` or `test_arclength_local_refine.py`.
- Use small synthetic inputs for selector, config, and output tests to avoid large video artifacts in version control.

## Commit & Pull Request Guidelines

- Recent history uses short Korean noun phrases, for example `클러스터 시각화 자동 선택 추가` and `README 최신 구조 반영`.
- Keep commits concise and scoped to one change.
- Pull requests should include the purpose, affected configs or modules, validation command, output path, and screenshots or links for visualization changes.

## Security & Configuration Tips

- Do not commit generated artifacts, large videos, local virtual environments, or machine-specific paths.
- Keep reusable behavior in YAML configs and code defaults rather than editing local-only scripts.
