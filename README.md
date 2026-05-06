# KS Video Latent Project

## 개요

영상에서 후보 프레임을 추출해 autoencoder를 학습하고, 전체 영상 프레임의 latent trajectory를 기준으로 keyframe을 선택하는 프로젝트입니다. 현재 학습 파이프라인은 Strategy 패턴으로 분리되어 모델, 선택 알고리즘, optimizer, loss, visualizer를 config에서 교체할 수 있습니다.

## 폴더 구조

| 경로 | 설명 |
|---|---|
| `src/keyframe_selection.py` | CLI 엔트리포인트 |
| `src/keyframe_pipeline/cli.py` | 학습/선택/저장 파이프라인 조립 코드 |
| `src/keyframe_pipeline/config.py` | YAML 로딩, config dataclass, 검증 |
| `src/keyframe_pipeline/models/` | autoencoder 모델 구현과 동적 로딩 factory |
| `src/keyframe_pipeline/optimizers/` | optimizer 구현과 동적 로딩 factory |
| `src/keyframe_pipeline/losses/` | loss 구현과 동적 로딩 factory |
| `src/keyframe_pipeline/selectors/` | keyframe 선택 구현과 동적 로딩 factory |
| `src/keyframe_pipeline/trainer.py` | 학습 루프 |
| `src/keyframe_pipeline/video.py` | 프레임 추출, 전처리, 전체 프레임 latent encoding |
| `src/keyframe_pipeline/visualizers/` | 시각화 구현과 동적 로딩 factory |
| `src/keyframe_pipeline/outputs.py` | CSV, checkpoint, NPZ, metrics 저장 |
| `src/video_split.py` | 별도 프레임 dataset 생성 유틸리티 |
| `configs/*.yaml` | 데이터셋/전략/학습/출력 설정 |
| `data/videos/` | 입력 비디오 |
| `artifacts/keyframe_selection/` | keyframe selection 결과 |

## 실행

conda base가 `python`을 덮는 환경에서는 아래 래퍼를 사용합니다.

```bash
bash scripts/run_keyframe_selection.sh --config configs/drone_rgb.yaml
```

## Config 구조

| section | 제어 대상 |
|---|---|
| `video` | 입력 비디오, 후보 프레임 수, 이미지 크기, 색상 모드 |
| `selection` | 선택 구현의 `module`/`class_name`, local refinement 세부값 |
| `model` | 모델 구현의 `module`/`class_name`, latent 차원, 모델 생성 kwargs |
| `optimizer` | optimizer 구현의 `module`/`class_name`, learning rate, weight decay, momentum |
| `loss` | loss 구현의 `module`/`class_name`, loss kwargs |
| `train` | epoch, batch size, seed, device |
| `output` | 결과 저장 위치와 파일명 |
| `visualization` | visualizer 구현의 `module`/`class_name`, 표시 옵션 |

## 현재 등록된 전략

| 모듈 | 기본 module/class | 설명 |
|---|---|---|
| model | `keyframe_pipeline.models.conv_autoencoder.ConvAutoEncoder` | 기존 CNN autoencoder |
| optimizer | `keyframe_pipeline.optimizers.torch_optimizers.AdamOptimizerStrategy` | Adam optimizer |
| loss | `keyframe_pipeline.losses.basic.MSELossStrategy` | MSE reconstruction loss |
| selection | `keyframe_pipeline.selectors.arclength_local_refine.ArclengthLocalRefineSelectionStrategy` | 기존 latent arclength 초기 선택 + local variance refinement |
| visualization | `keyframe_pipeline.visualizers.plotly_latent.PlotlyLatentVisualizationStrategy` | Plotly latent HTML |

## 산출물

| 파일 | 설명 |
|---|---|
| `artifacts/keyframe_selection/<run>/selected_frames.csv` | 선택된 프레임 번호와 latent 거리 |
| `artifacts/keyframe_selection/<run>/selected_frames/*.png` | 최종 선택 프레임 이미지 |
| `artifacts/keyframe_selection/<run>/uniform_frames/*.png` | 비교용 균등 샘플링 이미지 |
| `artifacts/keyframe_selection/<run>/frame_index_comparison.png` | 균등 샘플링과 keyframe 선택 비교 plot |
| `artifacts/keyframe_selection/<run>/all_frame_latents.npz` | 전체 영상 프레임 latent |
| `artifacts/keyframe_selection/<run>/latent_space.html` | latent 공간과 선택 경로 HTML |
| `artifacts/keyframe_selection/<run>/autoencoder.pt` | 학습된 autoencoder checkpoint |
| `artifacts/keyframe_selection/<run>/selection_metrics.json` | 선택 결과와 전략 메타데이터 |

## 교체 예시

새 모델을 추가하려면 `src/keyframe_pipeline/models/model2.py`에 `torch.nn.Module` 클래스를 만들고 `encode(inputs)`와 `forward(inputs)`를 구현합니다.

```yaml
model:
  name: model2
  module: keyframe_pipeline.models.model2
  class_name: Model2AutoEncoder
  latent_dim: 3
  kwargs:
    hidden_channels: 64
```

다른 하위 모듈도 같은 방식으로 교체합니다.

| 교체 대상 | 새 파일 예시 | class 요구사항 |
|---|---|---|
| 모델 | `models/model2.py` | `torch.nn.Module`, `encode(inputs)` |
| optimizer | `optimizers/adamw_custom.py` | `build(model, config)`가 `torch.optim.Optimizer` 반환 |
| loss | `losses/perceptual.py` | `build(config)`가 `torch.nn.Module` 반환 |
| 선택 알고리즘 | `selectors/my_selector.py` | `select(latents, config)`가 `SelectionResult` 반환 |
| 시각화 | `visualizers/my_visualizer.py` | `save(output_path, latents, frame_indices, timestamps_sec, selected_candidate_orders, config)` 구현 |
