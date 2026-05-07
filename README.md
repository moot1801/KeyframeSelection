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
| visualization | `keyframe_pipeline.visualizers.plotly_latent_controls.PlotlyLatentControlsVisualizationStrategy` | 체크박스 제어/통계 패널/linear view 포함 Plotly latent HTML |

## 자동 keyframe 개수 선택

기본 동작은 `selection.num_frames`에 지정한 개수를 그대로 사용합니다. 자동 선택은 `selection.kwargs.auto_num_frames.enabled`를 `true`로 설정했을 때만 활성화됩니다.

```yaml
selection:
  num_frames: 350
  kwargs:
    auto_num_frames:
      enabled: true
      method: max_k_under_cv
      min_frames: 50
      max_frames: 500
      cv_threshold: 0.20
      search_step: 10
      refine_candidates: false
```

자동 선택은 selected frame 사이 latent 거리의 `CV = 표준편차 / 평균`이 `cv_threshold` 이하인 후보 중 가장 큰 `k`를 선택합니다. `refine_candidates: true`로 설정하면 각 후보 `k` 평가 때도 local refinement를 수행하지만 실행 시간이 늘어납니다.

## 산출물

`output.output_dir`가 이미 존재하면 기존 결과를 덮어쓰지 않고 `<name>_001`, `<name>_002` 형식의 새 폴더를 생성합니다.

| 파일 | 설명 |
|---|---|
| `selected_frames.csv` | 선택된 프레임 순서, 원본 frame index, timestamp, latent 거리, cluster 분기, 이미지 경로 |
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

`PlotlyLatentControlsVisualizationStrategy`를 사용하면 HTML 오른쪽 패널에서 아래 레이어를 체크박스로 켜고 끌 수 있습니다.

| 레이어 | 설명 |
|---|---|
| Candidate frames | 전체 영상 프레임 latent 점 |
| Selected path | 선택 프레임 연결 경로 |
| Selected frames | 최종 선택 프레임 점 |
| Selected labels | 선택 프레임 label |
| Cluster endpoints | cluster별 시작/종료 선택 프레임 표시 |
| Candidate order | candidate order label |
| Linear latent view | latent trajectory 진행을 시작~종료 직선 위에 펼친 보조 view |

통계 패널에는 표시 중인 frame point 수, 선택 프레임 수, 선택 프레임 사이 latent L2 거리 평균/분산/표준편차, frame/time 범위가 표시됩니다. HTML 컨트롤에서 selected frame 점 크기를 조절하고, candidate order, frame index, selection order로 노드를 검색할 수 있습니다.

선택 프레임 사이 latent L2 거리는 먼저 selected 거리 기준으로 판단 대상 거리를 제외한 좌우 4개 주변 거리의 `local median + 3.5 * MAD scale`, 전체 거리의 `global 95 percentile`, 주변 median 대비 `1.5x` 비율 조건을 모두 만족할 때 분기 후보로 취급됩니다. 이후 해당 구간 주변 candidate adjacent 거리의 `local median + 3.5 * MAD scale`로 기대 거리를 계산하고, selected 거리가 이 기대 거리의 `2.0x` 이상일 때만 cluster 분기 후보로 유지합니다. candidate 흐름 자체가 sparse한 구간은 기대 거리가 커지므로 분할 기준이 완화됩니다. 후보 분기 후에는 최소 cluster 크기 3개를 만족하지 않는 작은 cluster가 생기지 않도록 가까운 분기를 병합합니다. `latent_space.html`에서는 분리된 selected cluster가 서로 다른 색으로 표시되고, `selected_frames.csv`, `all_frame_latents.npz`, `selection_metrics.json`에도 cluster 정보가 저장됩니다.

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
