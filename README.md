# Panorama Video Stitching Pipeline

축구장 좌/우 카메라 영상을 파노라마 영상으로 스티칭하는 파이프라인입니다.

## 📦 설치

```bash
pip install opencv-python numpy tqdm
```

FFmpeg (비디오 인코딩용, 선택사항):
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html 에서 다운로드
```

## 🚀 사용법

### 기본 사용법

```bash
python stitch_video.py \
    --left ./left_sync \
    --right ./right_sync \
    --calib calibration.json \
    --output ./output
```

### 전체 옵션

```bash
python stitch_video.py \
    --left ./left_sync \          # 좌측 카메라 프레임 폴더
    --right ./right_sync \        # 우측 카메라 프레임 폴더
    --calib calibration.json \    # 캘리브레이션 파일
    --output ./output \           # 출력 폴더
    --method cylindrical \        # 스티칭 방식 (cylindrical / planar)
    --workers 4 \                 # 병렬 처리 워커 수
    --fps 30                      # 출력 비디오 FPS
```

### Planar 방식 사용 (더 넓은 시야각)

```bash
python stitch_video.py \
    --left ./left_sync \
    --right ./right_sync \
    --calib calibration.json \
    --method planar
```

### 비디오만 생성 (이미 스티칭된 프레임이 있을 때)

```bash
python stitch_video.py \
    --video-only \
    --frames ./output/frames \
    --fps 30
```

## 📁 폴더 구조

```
project/
├── left_sync/              # 좌측 카메라 프레임
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
├── right_sync/             # 우측 카메라 프레임
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
├── calibration.json        # 캘리브레이션 데이터
├── stitch_video.py         # 메인 스크립트
└── output/                 # 출력 폴더 (자동 생성)
    ├── frames/             # 스티칭된 프레임들
    │   ├── stitched_frame_00001.jpg
    │   └── ...
    ├── panorama.mp4        # 출력 비디오
    └── panorama.web.mp4    # 웹 호환 비디오 (ffmpeg 필요)
```

## 🎬 비디오 뷰어

`video_viewer.html` 파일을 브라우저에서 열어 파노라마 비디오를 3D로 감상할 수 있습니다.

### 기능
- 🖱️ 드래그로 시점 회전
- 🔍 스크롤로 줌 인/아웃
- 3가지 투영 모드 (Cylinder, Flat Plane, Sphere)
- 곡률/거리 조절
- 재생 속도 조절

### 사용법
1. `video_viewer.html`을 브라우저에서 엽니다
2. 파노라마 비디오 파일을 드래그 앤 드롭합니다
3. 마우스로 시점을 조절하며 감상합니다

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
    "homography": [
        [h11, h12, h13],
        [h21, h22, h23],
        [h31, h32, h33]
    ]
}
```

## 📊 성능 참고

- 1920x1080 프레임 기준
- 8코어 CPU에서 약 10-15 FPS 처리 속도
- 메모리 사용량: 약 2-4GB

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
