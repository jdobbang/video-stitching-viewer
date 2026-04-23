"""
E2E Stitching Pipeline
======================
MOV 2개 입력 → 오디오 싱크 → 캘리브레이션 → 스티칭 → 뷰어 열기
"""

import os
import sys
import argparse
import time
import json
import itertools

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def step_sync(args):
    """[1/3] Audio Sync + Frame Export"""
    print(f"\n[1/3] Audio Sync + Frame Export")
    print(f"      Left:  {args.left}")
    print(f"      Right: {args.right}")
    t0 = time.time()

    from audio_sync import compute_sync_offset, save_sync_result, iter_synced_frames, get_frame_count

    # sync_verify를 output/{name}/sync_verify 에 저장
    sync_dir = os.path.join(args.output, "sync_verify")
    result = compute_sync_offset(
        args.left, args.right,
        fps=args.fps,
        out_dir=sync_dir,
        verbose=True,
    )
    save_sync_result(result, os.path.join(sync_dir, "sync_offset.txt"))

    # 총 프레임 수 계산 (tqdm 진행률 표시용)
    l_start = result["left_start"] - 1
    r_start = result["right_start"] - 1
    total_l = get_frame_count(args.left)
    total_r = get_frame_count(args.right)
    total_frames = min(total_l - l_start, total_r - r_start)
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)

    # 스트리밍 이터레이터 설정 (디스크 JPEG I/O 제거)
    frame_iter = iter_synced_frames(result, args.left, args.right,
                                    max_frames=args.max_frames)

    # 첫 프레임을 캘리브레이션용으로 보관 후 이터레이터에 재결합
    first_pair = next(frame_iter, None)
    if first_pair is not None:
        first_left, first_right = first_pair

        # 혼합 해상도 대응: 우측을 좌측 해상도에 맞춰 리사이즈
        target_hw = first_left.shape[:2]
        if first_right.shape[:2] != target_hw:
            th, tw = target_hw
            print(f"      Resolution mismatch: L={first_left.shape[1]}x{first_left.shape[0]} "
                  f"R={first_right.shape[1]}x{first_right.shape[0]} "
                  f"→ resizing R to L ({tw}x{th})")
            first_right = cv2.resize(first_right, (tw, th))

            def _normalize_right(it, th=th, tw=tw):
                for l, r in it:
                    if r.shape[:2] != (th, tw):
                        r = cv2.resize(r, (tw, th))
                    yield l, r
            frame_iter = _normalize_right(frame_iter)

        args._calib_left = first_left
        args._calib_right = first_right
        args._frame_iterator = itertools.chain([(first_left, first_right)], frame_iter)
        args._total_frames = total_frames
    else:
        args._calib_left = None
        args._calib_right = None
        args._frame_iterator = None
        args._total_frames = None

    args._sync_result = result

    print(f"\n      Offset: {result['offset']} frames | Confidence: {result['confidence']:.1f}")
    print(f"      Time: {fmt_time(time.time() - t0)}")
    return result


def step_calibrate(args):
    """[2/3] Auto Calibration"""
    print(f"\n[2/3] Auto Calibration (DISK + LightGlue)")
    t0 = time.time()

    # 메모리 이미지 사용 (스트리밍 경로) 또는 디스크 폴백
    if hasattr(args, '_calib_left') and args._calib_left is not None:
        left_img = args._calib_left
        right_img = args._calib_right
    else:
        left_frame_path = os.path.join(args.frames_dir, "left", "frame_000001.jpg")
        right_frame_path = os.path.join(args.frames_dir, "right", "frame_000001.jpg")
        if not os.path.exists(left_frame_path) or not os.path.exists(right_frame_path):
            print(f"  ERROR: Synced frames not found at {args.frames_dir}")
            sys.exit(1)
        left_img = cv2.imread(left_frame_path)
        right_img = cv2.imread(right_frame_path)

    from auto_calibrate import detect_and_match_lightglue, visualize

    print(f"      Image: {left_img.shape[1]}x{left_img.shape[0]}")
    print(f"      Matching (DISK + LightGlue)...")

    point_pairs, H = detect_and_match_lightglue(left_img, right_img)
    if not point_pairs:
        print("\n  ERROR: Matching failed")
        sys.exit(1)

    calib = {
        "point_pairs": point_pairs,
        "image_size": [left_img.shape[1], left_img.shape[0]],
        "method": "lightglue",
    }
    if args.left_focal is not None:
        calib["left_focal_mm"] = args.left_focal
    if args.right_focal is not None:
        calib["right_focal_mm"] = args.right_focal
    if H is not None:
        calib["homography"] = H.tolist()

    os.makedirs(os.path.dirname(args.calib), exist_ok=True)
    with open(args.calib, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    # 매칭 시각화 저장
    vis_path = os.path.join(os.path.dirname(args.calib), "matches_vis.jpg")
    visualize(left_img, right_img, point_pairs, vis_path)

    print(f"      Output: {args.calib} ({len(point_pairs)} pairs)")
    print(f"      Time: {fmt_time(time.time() - t0)}")


def step_stitch(args):
    """[3/3] Panorama Stitching"""
    print(f"\n[3/3] Panorama Stitching ({args.method})")
    t0 = time.time()

    from stitch_video import stitch_from_iterator, batch_stitch, create_video

    # 좌측 MOV 오디오를 최종 비디오에 포함
    audio_src = args.left if hasattr(args, 'left') and args.left else None

    pixel_match = not args.no_pixel_match
    lab_match = not args.no_lab_match
    multi_band = not args.no_multi_band
    if not pixel_match:
        print(f"      Pixel equalization: DISABLED (raw warped blending)")
    elif not lab_match:
        print(f"      Pixel equalization: BGR gain only (Lab a/b disabled)")
    print(f"      Blending: {'Multi-band (Laplacian pyramid)' if multi_band else 'Linear alpha'}")

    # 스트리밍 경로: 메모리 이터레이터 사용
    if hasattr(args, '_frame_iterator') and args._frame_iterator is not None:
        total = getattr(args, '_total_frames', None)
        stitch_from_iterator(
            args._frame_iterator, args.calib, args.output,
            method=args.method, focal_weight='auto',
            fps=int(args.fps) if args.fps else 30,
            total_frames=total,
            audio_source=audio_src,
            lab_match=lab_match,
            pixel_match=pixel_match,
            multi_band=multi_band,
        )
    else:
        # 디스크 기반 폴백 (--skip-sync 등)
        fw = 'auto'
        frames_out = os.path.join(args.output, "frames")
        batch_stitch(
            os.path.join(args.frames_dir, "left"),
            os.path.join(args.frames_dir, "right"),
            args.calib, frames_out,
            method=args.method, num_workers=args.workers,
            focal_weight=fw,
            lab_match=lab_match,
            pixel_match=pixel_match,
            multi_band=multi_band,
        )
        if args.fps:
            create_video(
                frames_out,
                os.path.join(args.output, "panorama.mp4"),
                fps=int(args.fps),
                audio_source=audio_src,
            )

    print(f"      Output: {args.output}")
    print(f"      Time: {fmt_time(time.time() - t0)}")


def main():
    parser = argparse.ArgumentParser(
        description="E2E Stitching Pipeline: MOV → Sync → Calibrate → Stitch"
    )
    parser.add_argument("pair", nargs="?", default=None,
                        help="Pair folder under asset/ (e.g. 0, 1, 2)")
    parser.add_argument("--left", default=None, help="Left camera MOV file")
    parser.add_argument("--right", default=None, help="Right camera MOV file")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--method", default="cylindrical", choices=["cylindrical", "planar"],
                        help="Stitching method (default: cylindrical)")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS override (auto-detect if omitted)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Stitching parallel workers")
    parser.add_argument("--frames-dir", default=None, help="Synced frames directory")
    parser.add_argument("--calib", default=None, help="Calibration JSON path")

    parser.add_argument("--left-focal", type=float, default=None,
                        help="좌측 카메라 35mm 환산 focal length (mm, overrides focal.json)")
    parser.add_argument("--right-focal", type=float, default=None,
                        help="우측 카메라 35mm 환산 focal length (mm, overrides focal.json)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to extract (default: all)")

    parser.add_argument("--no-pixel-match", action="store_true",
                        help="픽셀 이퀄라이징 전체 비활성화 (BGR 게인 + Lab a/b 모두 끔)")
    parser.add_argument("--no-lab-match", action="store_true",
                        help="Lab a/b 색상 매칭만 비활성화 (BGR 게인은 유지)")
    parser.add_argument("--no-multi-band", action="store_true",
                        help="Multi-band blending 비활성화 (기본: 활성화). 끄면 선형 알파 블렌딩 사용")

    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip audio sync (use existing frames)")
    parser.add_argument("--skip-calib", action="store_true",
                        help="Skip calibration (use existing JSON)")
    parser.add_argument("--skip-stitch", action="store_true",
                        help="Skip stitching")

    args = parser.parse_args()

    # pair 이름 결정: positional arg 또는 --left 파일명에서 추출
    if args.pair is not None:
        video_name = args.pair
    elif args.left is not None:
        video_name = os.path.splitext(os.path.basename(args.left))[0]
        video_name = video_name.replace("_left", "").replace("_right", "")
    else:
        parser.error("pair 번호 또는 --left 경로를 지정하세요. 예: python pipeline.py 1")

    # asset/{name}/ 폴더에서 영상 + focal.json 자동 로드
    pair_dir = os.path.join(BASE_DIR, "asset", video_name)
    if args.left is None:
        args.left = os.path.join(pair_dir, "left.MOV")
    if args.right is None:
        args.right = os.path.join(pair_dir, "right.MOV")

    # focal.json 자동 로드 (CLI 인자가 없을 때)
    focal_path = os.path.join(pair_dir, "focal.json")
    if os.path.exists(focal_path):
        with open(focal_path, "r") as f:
            focal_cfg = json.load(f)
        if args.left_focal is None and focal_cfg.get("left_focal_mm") is not None:
            args.left_focal = focal_cfg["left_focal_mm"]
        if args.right_focal is None and focal_cfg.get("right_focal_mm") is not None:
            args.right_focal = focal_cfg["right_focal_mm"]

    has_focal = args.left_focal is not None

    # 미지정 경로를 비디오 이름 기반으로 자동 생성
    if args.frames_dir is None:
        args.frames_dir = os.path.join(BASE_DIR, "frames_sync", video_name)
    if args.calib is None:
        calib_name = "calib_spec.json" if has_focal else "calib.json"
        args.calib = os.path.join(BASE_DIR, "calibrations", video_name, calib_name)
    if args.output is None:
        out_name = f"{video_name}_spec" if has_focal else video_name
        args.output = os.path.join(BASE_DIR, "output", out_name)

    print("=" * 60)
    print("  E2E Stitching Pipeline")
    print("=" * 60)
    t_total = time.time()

    # [1] Sync
    if not args.skip_sync:
        step_sync(args)
    else:
        print(f"\n[1/3] Audio Sync — SKIPPED")

    # [2] Calibrate
    if not args.skip_calib:
        step_calibrate(args)
    else:
        print(f"\n[2/3] Calibration — SKIPPED")

    # [3] Stitch
    if not args.skip_stitch:
        step_stitch(args)
    else:
        print(f"\n[3/3] Stitching — SKIPPED")

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete! Total time: {fmt_time(time.time() - t_total)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
