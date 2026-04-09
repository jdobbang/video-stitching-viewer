# CLAUDE.md

## 프로젝트 개요
축구장 좌/우 카메라 영상을 파노라마로 스티칭하고, 3D 뷰어로 감상 + 이벤트 메모를 남기는 시스템.

## 핵심 파일 구조

### 파이프라인 (Python)
- `pipeline.py` — E2E 오케스트레이션: sync → calibrate → stitch
- `audio_sync.py` — 오디오 크로스코릴레이션으로 좌/우 영상 동기화
- `auto_calibrate.py` — DISK + LightGlue 자동 캘리브레이션
- `stitch_video.py` — PanoramaStitcher 클래스 + create_video (FFmpeg mux)
- `yolo26.py` — YOLO 객체 검출 모델

### 뷰어 (HTML, `VIEW TEST/` 폴더)
- `viewer_expert.html` — **메인 뷰어** (활발히 개발 중)
- `tactics_expert.html` — 전술 미니맵 + 캘리브레이션 (기능 분리됨)
- `base.html` — 기본 뷰어 (구버전)

### 자산
- `asset/` — 소스 MOV, focal.json, YOLO 모델 (.onnx, .pt)
- `calibrations/` — 캘리브레이션 JSON 결과
- `output/` — 스티칭 결과 (panorama.mp4, panorama.web.mp4)

## viewer_expert.html 현재 상태

### 코어 기능
- Three.js 파노라마 렌더링 (Cylinder/Sphere 투영 모드)
- 비디오 드래그앤드롭 로드 (+ JSON 북마크 파일 함께 드롭 가능)
- 재생/정지, 속도 조절, 프로그레스바 (하단 풀와이드)
- 마우스 드래그 시점 이동, 클릭 이동, 스크롤 줌
- 키보드: AWDS 이동, QE 회전, RF 줌, Space 재생, 화살표 5초 이동
- 관성 시스템 (부드러운 가속/감속)
- FOV 슬라이더, 비디오 회전, 스크린샷(P), 풀스크린(F11)
- 프레임 단위 이동 (,/.)

### 이벤트 메모 시스템
- 태그: ⚽골, ⚔️공격, 🛡️수비, 📋피드백, 📝메모 (색상별 구분)
- 우측 패널: 태그 필터 (클릭=단독, Ctrl+클릭=토글), 시간순 리스트
- 재생바 마커: 이모지 + 컬러바, 클릭 시 해당 시간 이동
- 더블클릭으로 메모 인라인 수정
- JSON 저장/불러오기 (File System Access API, 자동 덮어쓰기)
- M키: 일시정지 + 메모 입력 포커스

### UI
- 모든 패널 토글(접기/펼치기) 가능
- 오디오 파형: 비디오에 오디오 있으면 재생바 위에 표시
- 코덱 미지원 안내 (mp4v → .web.mp4 사용 안내)

### 모바일 반응형 (768px 이하)
- 상단 탭바로 패널 전환 (👁️📌🎛️🎥ℹ️)
- 하단 시트(bottom sheet) 레이아웃
- 핀치 줌 → FOV 줌 매핑
- 터치 드래그 재생바 seek
- 44px 최소 터치 타겟, 큰 폰트
- autoplay muted → 첫 터치 시 unmute
- 탭 클릭 시 PTZ 이동 비활성화 (드래그만 허용)
- 플로팅 컨트롤 버튼 (⏮📸⏭)

## 파이프라인 주요 사항

### 오디오 Mux
- `create_video()`에서 `audio_source` 파라미터로 좌측 MOV 오디오를 AAC 192kbps로 합침
- `pipeline.py`가 `args.left`를 자동 전달
- `-shortest` 플래그로 비디오/오디오 길이 맞춤

### 스트리밍 구조
- `stitch_from_iterator()`가 이미 메모리 이터레이터 기반 (디스크 I/O 최소화)
- 720p CPU ~37fps, 1080p는 GPU(CUDA) 필요
- 라이브 스트리밍 가능성 있음 (캘리브레이션 1회 → stitch 루프만 실시간)

### 코덱
- `panorama.mp4` — OpenCV mp4v (브라우저 재생 불가)
- `panorama.web.mp4` — H.264 libx264 CRF 23 (브라우저 재생 가능, 이것만 사용)

## 코딩 규칙
- 한국어 UI/주석, 변수명은 영어
- 단일 HTML 파일 (외부 의존성: Three.js CDN만)
- File System Access API 사용 시 반드시 fallback 다운로드 포함
- 모바일 수정 시 `_isMobile` 플래그로 분기

## 외부 접속 (ngrok)
```bash
# 1. 로컬 서버
cd "VIEW TEST"
python -m http.server 8080 --bind 0.0.0.0

# 2. ngrok 터널 (외부 접속용)
ngrok http --url=perplexingly-fortuitous-hyman.ngrok-free.dev 8080
# → https://perplexingly-fortuitous-hyman.ngrok-free.dev/viewer_expert.html
```
- ngrok 설치 경로: `C:\Users\sqlabs_ai\AppData\Local\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe`
- authtoken 설정 완료 (ngrok.yml)
- 같은 Wi-Fi: `http://10.137.50.37:8080/viewer_expert.html`

## 미구현 / 향후 계획
- 북마크 간 이전/다음 이동 버튼 (재생바에 ◀▶)
- A-B 구간 반복 재생
- 키보드 단축키 도움말 오버레이 (? 키)
- 볼륨 컨트롤 (재생바)
- 북마크 스크린샷 썸네일
- 구간 북마크 (시작~끝 범위)
- 하이라이트 자동 재생 (북마크 연속 재생)
- 이벤트 텍스트 내보내기 (카톡 공유용)
- 라이브 스트리밍 프로토타입 (RTMP 입력 → 실시간 스티칭 → WebRTC 출력)

## Git
- 메인 브랜치: main
- 원격: https://github.com/jdobbang/video-stitching-viewer
- 사용자: jdobbang

## 최근 작업 히스토리 (2026-04-09)
1. soccer_tactics_fixed.html에서 캘리브레이션/전술/YOLO/GIF 기능 분리
2. viewer_expert.html 생성 — 순수 뷰어 전용 (470줄 → 현재 ~1700줄)
3. 이벤트 메모 시스템 추가 (태그, 필터, 재생바 마커, JSON 저장, 더블클릭 수정)
4. 하단 풀와이드 재생바 + 패널 토글 + 오디오 파형
5. 파이프라인 오디오 mux 추가 (좌측 MOV → AAC → .web.mp4)
6. 풀 모바일 반응형 (탭바, 핀치줌, 터치 seek, 44px 터치타겟)
7. ngrok 설정 완료 — 외부에서 폰 접속 가능
