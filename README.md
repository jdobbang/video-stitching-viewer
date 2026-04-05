# Panorama Video Stitching Pipeline

축구장 좌/우 카메라 영상을 파노라마 영상으로 스티칭하는 E2E 파이프라인입니다.

![Panorama Video Viewer](asset/image.png)

## 📦 설치

```bash
pip install opencv-python numpy tqdm torch torchvision
pip install kornia  # DISK + LightGlue 매칭용
```

FFmpeg (비디오 인코딩용):
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html 에서 다운로드
```

## 🚀 사용법

### E2E 파이프라인 (권장)

MOV 2개 입력 → 오디오 싱크 → 자동 캘리브레이션 → 스티칭 → 뷰어 열기

```bash
python pipeline.py --left left.MOV --right right.MOV
```

#### 파이프라인 옵션

```bash
python pipeline.py \
    --left left.MOV \             # 좌측 카메라 영상
    --right right.MOV \           # 우측 카메라 영상
    --output ./output \           # 출력 폴더 (기본: output/)
    --method cylindrical \        # 스티칭 방식 (cylindrical / planar)
    --fps 30 \                    # FPS override (기본: 자동 감지)
    --workers 4 \                 # 병렬 처리 워커 수
    --left-focal 50 \             # 좌측 카메라 35mm 환산 focal length (mm)
    --right-focal 50 \            # 우측 카메라 35mm 환산 focal length (mm)
    --max-frames 1000             # 최대 추출 프레임 수 (기본: 전체)
```

> `--left-focal` / `--right-focal` 지정 시 출력 경로에 `_spec` 접미사가 자동 추가됩니다.

#### 단계 건너뛰기

```bash
# 이미 싱크된 프레임이 있을 때
python pipeline.py --left left.MOV --right right.MOV --skip-sync

# 이미 캘리브레이션이 있을 때
python pipeline.py --left left.MOV --right right.MOV --skip-calib

# 스티칭만 건너뛰기
python pipeline.py --left left.MOV --right right.MOV --skip-stitch
```

### 개별 스크립트 실행

#### 1. 오디오 싱크 (audio_sync.py)

좌/우 영상의 오디오를 분석해 시간 오프셋을 계산하고 동기화된 프레임을 추출합니다.

#### 2. 자동 캘리브레이션 (auto_calibrate.py)

DISK + LightGlue로 좌/우 이미지 간 대응점을 자동 매칭합니다.

```bash
python auto_calibrate.py \
    --left ./left_sync/frame_00001.jpg \
    --right ./right_sync/frame_00001.jpg \
    --output calibration_auto.json \
    --left-focal 50 \             # 선택: 35mm 환산 focal length
    --right-focal 50
```

#### 3. 스티칭 (stitch_video.py)

```bash
python stitch_video.py \
    --left ./left_sync \
    --right ./right_sync \
    --calib calibration.json \
    --output ./output \
    --method cylindrical \
    --focal-weight 0.5 \          # focal length 가중치 (0~1)
    --workers 4 \
    --fps 30
```

#### 비디오만 생성 (이미 스티칭된 프레임이 있을 때)

```bash
python stitch_video.py \
    --video-only \
    --frames ./output/frames \
    --fps 30
```

## 📁 폴더 구조

```
project/
├── pipeline.py                 # E2E 파이프라인
├── audio_sync.py               # 오디오 싱크
├── auto_calibrate.py           # 자동 캘리브레이션 (DISK + LightGlue)
├── stitch_video.py             # 파노라마 스티칭
├── video_viewer.html           # 3D 파노라마 뷰어
├── calibration_auto.json       # 자동 생성된 캘리브레이션
├── calib_pair*.json            # 페어별 캘리브레이션 데이터
├── left_sync/                  # 좌측 동기화 프레임
├── right_sync/                 # 우측 동기화 프레임
└── output/                     # 출력 폴더 (자동 생성)
    ├── frames/                 # 스티칭된 프레임들
    ├── panorama.mp4            # 출력 비디오
    └── panorama.web.mp4        # 웹 호환 비디오 (ffmpeg)
```

## 🎬 비디오 뷰어

`video_viewer.html` 파일을 브라우저에서 열어 파노라마 비디오를 3D로 감상할 수 있습니다.

### 기능
- 드래그로 시점 회전
- 스크롤로 줌 인/아웃
- 3가지 투영 모드 (Cylinder, Flat Plane, Sphere)
- 곡률/거리 조절
- 재생 속도 조절

## ⚙️ 스티칭 방식 비교

| 방식 | 장점 | 단점 |
|------|------|------|
| **cylindrical** | 외곽 왜곡 감소, 자연스러운 비율 | 시야각 좁음 |
| **planar** | 넓은 시야각, 빠른 처리 | 외곽 물체 확대 |

## 🔧 캘리브레이션 파일 형식

```json
{
    "point_pairs": [
        {"left": [x1, y1], "right": [x1, y1]},
        ...
    ],
    "image_size": [1920, 1080],
    "method": "lightglue",
    "homography": [[...], [...], [...]],
    "left_focal_mm": 50,
    "right_focal_mm": 50
}
```

## 🐛 문제 해결

### "No matching right frame" 경고
- 좌/우 프레임 파일명이 일치하는지 확인하세요
- 지원 패턴: `frame_*.jpg`, `*.jpg`, `*.png`

### 메모리 부족
- `--workers 1`로 단일 스레드 처리

### 비디오 재생 안됨
- FFmpeg로 웹 호환 버전 생성:
```bash
ffmpeg -i panorama.mp4 -c:v libx264 -crf 23 -pix_fmt yuv420p panorama_web.mp4
```

## 📝 라이선스

MIT License
