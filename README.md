# KS Video Latent Project

## 개요

영상에서 사용자가 지정한 개수만큼 프레임을 균등 추출하고, 이를 2D latent autoencoder로 학습한 뒤, 시간 흐름에 따른 latent trajectory를 시각화하는 프로젝트 구조입니다.

## 디렉토리 구조

| 경로 | 설명 |
|---|---|
| `KS/` | 로컬 가상환경 |
| `src/video_split.py` | 영상 프레임 추출 및 `.npz` 데이터셋 생성 |
| `src/2d_AE.py` | 2D latent autoencoder 학습 및 latent 저장 |
| `src/view_latent.py` | latent trajectory 시각화 |
| `src/3d_AE.py` | 3D latent autoencoder 학습 및 epoch별 3D snapshot 저장 |
| `src/3d_view_latent.py` | 3D latent trajectory를 인터랙티브 HTML로 시각화 |
| `data/frames/` | 분할 프레임과 전처리 데이터셋 저장 |
| `artifacts/models/` | 학습된 모델과 학습 로그 저장 |
| `artifacts/latents/` | latent 벡터 저장 |
| `artifacts/plots/` | latent 시각화 이미지 저장 |

## 실행 순서

1. 가상환경 활성화

```bash
source KS/bin/activate
```

2. 패키지 설치

```bash
pip install -r requirements.txt
```

3. 영상 프레임 추출 및 데이터셋 생성

```bash
python src/video_split.py \
  --video-path path/to/video.mp4 \
  --num-frames 120 \
  --image-size 64 \
  --color-mode rgb
```

4. 오토인코더 학습

```bash
python src/2d_AE.py \
  --dataset data/frames/video_120f_64px_rgb/dataset.npz \
  --epochs 100 \
  --batch-size 16
```

5. latent trajectory 시각화

```bash
python src/view_latent.py \
  --latents artifacts/latents/video_120f_64px_rgb_latents.npz \
  --annotate-every 10
```

6. 3D latent autoencoder 학습

```bash
python src/3d_AE.py \
  --dataset data/frames/video_120f_64px_rgb/dataset.npz \
  --epochs 100 \
  --batch-size 16 \
  --visualize-every 10 \
  --annotate-every 50
```

7. 3D latent trajectory 인터랙티브 시각화

```bash
python src/3d_view_latent.py \
  --latents artifacts/latents/video_120f_64px_rgb_3d_latents.npz \
  --annotate-every 25
```

8. YAML 기반 latent 거리 균등 프레임 선택

conda base가 `python`을 덮는 환경에서는 아래 래퍼를 사용합니다.

```bash
bash scripts/run_keyframe_selection.sh --config configs/keyframe_selection_example.yaml
```

## 산출물

| 파일 | 설명 |
|---|---|
| `data/frames/<dataset>/images/*.png` | 추출된 프레임 미리보기 |
| `data/frames/<dataset>/dataset.npz` | 오토인코더 입력용 프레임 텐서 |
| `data/frames/<dataset>/metadata.json` | 프레임 인덱스, 시간 정보 등 메타데이터 |
| `artifacts/models/<run>.pt` | 학습된 autoencoder 체크포인트 |
| `artifacts/models/<run>_history.json` | 학습 loss 이력 |
| `artifacts/latents/<run>_latents.npz` | 프레임별 2D latent 좌표 |
| `artifacts/plots/<run>_trajectory.png` | 시간 순서가 반영된 latent 시각화 |
| `artifacts/models/<run>_3d.pt` | 학습된 3D autoencoder 체크포인트 |
| `artifacts/latents/<run>_3d_latents.npz` | 프레임별 3D latent 좌표 |
| `artifacts/latents/<run>_3d_training/epoch_*.npz` | epoch별 3D latent snapshot |
| `artifacts/plots/<run>_3d_training/epoch_*.html` | 학습 중간 3D latent 인터랙티브 시각화 |
| `artifacts/plots/<run>_3d_trajectory.html` | 최종 3D latent 인터랙티브 시각화 |
| `artifacts/keyframe_selection/<run>/selected_frames.csv` | latent 거리 기준으로 선택된 실제 영상 프레임 번호 |
| `artifacts/keyframe_selection/<run>/selected_frames/*.png` | 최종 선택된 프레임 이미지 |
| `artifacts/keyframe_selection/<run>/uniform_frames/*.png` | 비교용 동일 개수 균등 분할 프레임 이미지 |
| `artifacts/keyframe_selection/<run>/frame_index_comparison.png` | 균등 분할과 latent keyframe selection의 원본 프레임 번호 비교 그래프 |
| `artifacts/keyframe_selection/<run>/all_frame_latents.npz` | 학습된 encoder로 계산한 전체 영상 프레임 latent |
| `artifacts/keyframe_selection/<run>/latent_space.html` | 전체 영상 프레임과 선택 경로 Plotly HTML |

## 참고

| 항목 | 내용 |
|---|---|
| 입력 프레임 형태 | `(N, C, H, W)` |
| latent 차원 | `2` 고정 |
| 기본 이미지 크기 | `64 x 64` |
| 모델 제약 | 입력 크기는 `16`으로 나누어 떨어져야 함 |
| 3D 시각화 | `plotly` 기반 HTML, 브라우저에서 회전/확대/hover 가능 |
| 설치 상태 | `numpy`, `opencv-python`, `matplotlib`, `torch`는 설치됨. `plotly`는 requirements에 추가되었고 별도 설치가 필요할 수 있음 |
