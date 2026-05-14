# KS Video Latent Project

## 개요

영상에서 후보 프레임을 추출해 autoencoder를 학습하고, 전체 영상 프레임의 latent trajectory를 기준으로 keyframe을 선택하는 프로젝트입니다. 파이프라인은 YAML 설정 중심으로 동작하며 모델, optimizer, loss, keyframe selector, visualizer를 `module`/`class_name`으로 교체할 수 있습니다.

현재 기본 선택 방식은 latent 경로 길이 기준 초기 선택 후, 선택 프레임 사이 latent L2 거리 분산을 줄이는 local refinement를 수행합니다.

## 실행

conda base가 `python`을 덮는 환경에서는 저장소에 포함된 래퍼를 사용합니다.

```bash
bash scripts/run_keyframe_selection.sh --config configs/drone_gray2.yaml
```

다른 설정은 `configs/*.yaml` 파일을 지정해 실행합니다.

```bash
bash scripts/run_keyframe_selection.sh --config configs/drone_rgb.yaml
bash scripts/run_keyframe_selection.sh --config configs/tino_gray.yaml
bash scripts/run_keyframe_selection.sh --config configs/tino_rgb.yaml
```

## 파이프라인

| 단계 | 처리 내용 | 주요 코드 |
|---|---|---|
| 1 | 영상에서 학습용 후보 프레임 균등 추출 | `src/keyframe_pipeline/video.py` |
| 2 | 후보 프레임으로 autoencoder 학습 | `src/keyframe_pipeline/trainer.py` |
| 3 | 학습된 encoder로 전체 영상 프레임 latent 추출 | `src/keyframe_pipeline/video.py` |
| 4 | latent 누적 경로 길이 기준 초기 keyframe 선택 | `src/keyframe_pipeline/selectors/arclength_local_refine.py` |
| 5 | 선택 프레임 간 latent L2 거리 분산 local refinement | `src/keyframe_pipeline/selectors/arclength_local_refine.py` |
| 6 | 선택/비교 프레임 이미지, CSV, plot, 영상 저장 | `src/keyframe_pipeline/cli.py` |
| 7 | checkpoint, latent NPZ, metrics JSON 저장 | `src/keyframe_pipeline/outputs.py` |
| 8 | latent HTML 시각화 저장 | `src/keyframe_pipeline/visualizers/` |

실행 중 터미널에는 후보 프레임 추출, 학습 설정, latent encoding 진행률, local refinement iteration별 분산 변화, 산출물 저장 경로가 출력됩니다.

## 폴더 구조

| 경로 | 설명 |
|---|---|
| `src/keyframe_selection.py` | CLI 엔트리포인트 |
| `src/keyframe_pipeline/cli.py` | 전체 파이프라인 조립 및 산출물 저장 흐름 |
| `src/keyframe_pipeline/config.py` | YAML 로딩, config dataclass, 검증 |
| `src/keyframe_pipeline/loading.py` | `module`/`class_name` 기반 동적 class 로딩 |
| `src/keyframe_pipeline/models/` | autoencoder 모델 구현과 factory |
| `src/keyframe_pipeline/optimizers/` | optimizer strategy 구현과 factory |
| `src/keyframe_pipeline/losses/` | loss strategy 구현과 factory |
| `src/keyframe_pipeline/selectors/` | keyframe 선택 strategy 구현과 검증 |
| `src/keyframe_pipeline/visualizers/` | Plotly HTML, timeline/plot 시각화 관련 코드 |
| `src/keyframe_pipeline/video.py` | OpenCV 기반 영상 처리, 프레임 export, timeline 비교 영상 생성 |
| `src/keyframe_pipeline/outputs.py` | CSV, checkpoint, NPZ, metrics 저장 |
| `src/video_split.py` | 별도 프레임 dataset 생성 유틸리티 |
| `src/latent_distance_report.py` | latent 거리 분석 유틸리티 |
| `src/latent_distance_insights.py` | latent 거리 분석 보조 유틸리티 |
| `configs/*.yaml` | 데이터셋/전략/학습/출력/시각화 설정 |
| `data/videos/` | 입력 비디오 위치 |
| `data/frames/` | 추출 프레임 dataset 위치 |
| `artifacts/keyframe_selection/` | 실행 결과 저장 위치 |
| `artifacts/models/`, `artifacts/latents/`, `artifacts/plots/` | 모델/latent/plot 보조 산출물 위치 |

## Config 구조

| section | 제어 대상 |
|---|---|
| `video` | 입력 비디오, 후보 프레임 수, 이미지 크기, 색상 모드 |
| `selection` | 선택 strategy, 선택 프레임 수, 거리 metric, endpoint 포함 여부, local refinement 설정 |
| `model` | 모델 strategy, latent 차원, 모델 생성 kwargs |
| `optimizer` | optimizer strategy, learning rate, weight decay, momentum |
| `loss` | loss strategy와 loss kwargs |
| `train` | epoch, batch size, seed, worker 수, device |
| `output` | 결과 저장 위치와 파일명 |
| `visualization` | HTML visualizer strategy와 표시 옵션 |

### 주요 설정 예시

```yaml
selection:
  name: arclength_local_refine
  module: keyframe_pipeline.selectors.arclength_local_refine
  class_name: ArclengthLocalRefineSelectionStrategy
  num_frames: 32
  distance_metric: l2
  include_endpoints: true
  local_refine_iterations: 3
  local_refine_window: 30

visualization:
  name: plotly_latent_controls
  module: keyframe_pipeline.visualizers.plotly_latent_controls
  class_name: PlotlyLatentControlsVisualizationStrategy
  dimensions: 3
  annotate_every: 5
  show_all_candidates: true
  show_selected_path: true
  kwargs:
    show_candidate_frames: true
    show_selected_path: true
    show_selected_frames: true
    show_selected_labels: true
    show_candidate_order: false
    show_linear_view: false
    candidate_order_every: 10
```

## 등록된 기본 전략

| 모듈 | 기본 module/class | 설명 |
|---|---|---|
| model | `keyframe_pipeline.models.conv_autoencoder.ConvAutoEncoder` | CNN autoencoder |
| optimizer | `keyframe_pipeline.optimizers.torch_optimizers.AdamOptimizerStrategy` | Adam optimizer |
| loss | `keyframe_pipeline.losses.basic.MSELossStrategy` | MSE reconstruction loss |
| selection | `keyframe_pipeline.selectors.arclength_local_refine.ArclengthLocalRefineSelectionStrategy` | latent arclength 초기 선택 + local variance refinement |
| visualization | `keyframe_pipeline.visualizers.plotly_latent.PlotlyLatentVisualizationStrategy` | 기본 Plotly latent HTML |
| visualization | `keyframe_pipeline.visualizers.plotly_latent_controls.PlotlyLatentControlsVisualizationStrategy` | Plotly latent HTML + layer/cluster controls |

## 자동 keyframe 개수 선택

기본 동작은 `selection.num_frames`에 지정한 개수를 그대로 사용합니다. 자동 선택은 `selection.kwargs.auto_num_frames.enabled`를 `true`로 설정했을 때만 활성화됩니다.

```yaml
selection:
  num_frames: 350
  kwargs:
    auto_num_frames:
      enabled: true
      frame_limit_enabled: false
      target_gap_ratio: 10.0
      soft_cv_target: 0.25
      density_weight: 1.0
      cv_weight: 0.5
      search_step: 10
      refine_candidates: false
    precluster:
      enabled: true
      global_percentile: 99.0
      local_window: 5
      local_mad_multiplier: 3.5
      min_local_median_ratio: 8.0
      min_local_threshold_ratio: 3.0
      max_neighbor_high_count: 1
      min_cluster_frames: 30
      min_frames_per_cluster: 2
```

자동 선택은 selected frame 사이 평균 latent 거리를 cluster 내부 candidate adjacent 거리의 median으로 나눈 `gap_ratio`가 `target_gap_ratio`에 가까울수록 낮은 점수를 주고, `CV = 표준편차 / 평균`이 `soft_cv_target`을 초과한 만큼 penalty를 더합니다. 점수는 `density_weight * |gap_ratio-target|/target + cv_weight * max(0, CV-soft_cv_target)`입니다. `frame_limit_enabled: false`이면 탐색 상한은 전체 candidate 수의 `1/5`이고, `true`이면 `min_frames`와 `max_frames` 범위 안에서만 탐색합니다. `refine_candidates: true`로 설정하면 각 후보 `k` 평가 때도 local refinement를 수행하지만 실행 시간이 늘어납니다.

`precluster.enabled: true`이면 전체 candidate adjacent latent distance에서 scene jump를 먼저 감지합니다. 후보 jump는 전체 분포 상위 outlier이면서 주변 median과 robust local threshold를 크게 초과해야 하며, 주변에도 큰 값이 연속되는 속도 변화 구간은 제외됩니다. precluster가 2개 이상 cluster를 만들면 cluster별로 프레임을 선택하고, 자동 `k` 평가의 CV/분산은 cluster boundary 거리를 제외한 cluster 내부 selected 거리로 계산합니다.

## 산출물

`output.output_dir`가 이미 존재하면 기존 결과를 덮어쓰지 않고 `<name>_001`, `<name>_002` 형식의 새 폴더를 생성합니다.

| 파일 | 설명 |
|---|---|
| `selected_frames.csv` | 선택된 프레임 순서, 원본 frame index, timestamp, latent 거리, precluster 정보, 이미지 경로 |
| `selected_frames/*.png` | 최종 선택 프레임 이미지 |
| `uniform_frames/*.png` | 비교용 균등 샘플링 프레임 이미지 |
| `frame_index_comparison.png` | selection order 기준 uniform/keyframe frame index 비교 plot |
| `timeline_comparison.mp4` | 왼쪽 원본 영상 + 오른쪽 frame index timeline 그래프 + 현재 위치 bar |
| `selected_vs_uniform.mp4` | 왼쪽 KS 선택 프레임 + 오른쪽 균등 샘플링 프레임을 일정 간격으로 넘기는 비교 영상 |
| `all_frame_latents.npz` | 전체 영상 프레임 latent, frame index, timestamp, 선택 결과 |
| `latent_space.html` | latent 공간, 선택 경로, label, 통계 패널 HTML |
| `autoencoder.pt` | 학습된 autoencoder checkpoint |
| `selection_metrics.json` | 선택 결과와 학습/전략 메타데이터 |

## 시각화

### `latent_space.html`

`latent_space.html`은 전체 candidate latent와 최종 selected path를 표시합니다. cluster 분리는 selected frame 거리 후처리가 아니라 `precluster` 결과만 기준으로 판단하며, 해당 정보는 `selected_frames.csv`, `all_frame_latents.npz`, `selection_metrics.json`에 저장됩니다.

`plotly_latent_controls`는 HTML 우측 패널에서 다음 기능을 제공합니다.

| 기능 | 설명 |
|---|---|
| layer toggle | candidate frames, selected frames, frame path 표시/숨김 |
| cluster toggle | precluster별 checkbox로 특정 cluster만 표시/숨김 |
| cluster endpoints | 각 cluster 시작점은 초록 diamond, 끝점은 빨간 x marker로 표시 |
| point size | candidate/selected frame marker 크기 slider 조절 |
| stats panel | trajectory 거리, selected 거리, cluster별 path/selected 통계 |

### `timeline_comparison.mp4`

원본 영상과 선택 결과를 함께 확인하기 위한 MP4 산출물입니다.

| 영역 | 설명 |
|---|---|
| 왼쪽 | 원본 영상 프레임 |
| 오른쪽 | x축이 원본 frame index인 timeline 그래프 |
| selected row | keyframe selector가 고른 frame index |
| uniform row | 균등 샘플링 비교 frame index |
| current bar | 현재 재생 중인 원본 frame index 위치 |

### `selected_vs_uniform.mp4`

KS 선택 프레임 집합과 균등 샘플링 프레임 집합을 같은 순서로 나란히 보여주는 MP4 산출물입니다.

| 영역 | 설명 |
|---|---|
| 왼쪽 | KS selector가 선택한 프레임 |
| 오른쪽 | 같은 order의 균등 샘플링 프레임 |
| 전환 간격 | `output.selected_vs_uniform_interval_sec`, 기본값 `0.2`초 |

## 선택 알고리즘

현재 selector는 다음 순서로 동작합니다.

| 단계 | 설명 |
|---|---|
| 1 | 전체 영상 latent `z_i`에 대해 인접 프레임 L2 거리 계산 |
| 2 | 인접 거리 누적으로 latent trajectory arclength 계산 |
| 3 | 전체 arclength를 기준으로 초기 keyframe 집합 선택 |
| 4 | 각 선택 프레임을 주변 후보로 바꿔보며 선택 프레임 간 직접 L2 거리 분산 계산 |
| 5 | 분산이 줄어드는 교체만 채택 |
| 6 | `local_refine_iterations`만큼 반복하거나 개선이 없으면 조기 종료 |

주의할 점은 `Linear latent view`는 시각화용 보조 view이며, local refinement가 직접 최소화하는 값은 원래 latent space에서 선택 프레임 간 직접 L2 거리 분산입니다.

## 확장 방법

새 구현을 추가한 뒤 YAML의 `module`/`class_name`만 바꾸면 됩니다.

| 교체 대상 | 새 파일 예시 | class 요구사항 |
|---|---|---|
| 모델 | `src/keyframe_pipeline/models/model2.py` | `torch.nn.Module`, `encode(inputs)`, `forward(inputs)` |
| optimizer | `src/keyframe_pipeline/optimizers/adamw_custom.py` | `build(model, config)`가 `torch.optim.Optimizer` 반환 |
| loss | `src/keyframe_pipeline/losses/perceptual.py` | `build(config)`가 `torch.nn.Module` 반환 |
| 선택 알고리즘 | `src/keyframe_pipeline/selectors/my_selector.py` | `select(latents, config)`가 `SelectionResult` 형태 결과 반환 |
| 시각화 | `src/keyframe_pipeline/visualizers/my_visualizer.py` | `save(output_path, latents, frame_indices, timestamps_sec, selected_candidate_orders, config)` 구현 |

예시:

```yaml
model:
  name: model2
  module: keyframe_pipeline.models.model2
  class_name: Model2AutoEncoder
  latent_dim: 3
  kwargs:
    hidden_channels: 64
```

## 의존성

```bash
pip install -r requirements.txt
```

주요 의존성은 PyTorch, OpenCV, NumPy, Matplotlib, Plotly입니다.
