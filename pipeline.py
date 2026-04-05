"""
E2E Stitching Pipeline
======================
MOV 2개 입력 → 오디오 싱크 → 캘리브레이션 → 스티칭 → 뷰어 열기
"""

import os
import sys
import subprocess
import argparse
import time
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def run_step(name, cmd):
    """subprocess 실행 + 실시간 stdout 출력"""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(f"      {line}", end="")
    proc.wait()
    if proc.returncode != 0:
        print(f"\n  ERROR: {name} failed (exit code {proc.returncode})")
        sys.exit(1)


def step_sync(args):
    """[1/4] Audio Sync + Frame Export"""
    print(f"\n[1/3] Audio Sync + Frame Export")
    print(f"      Left:  {args.left}")
    print(f"      Right: {args.right}")
    t0 = time.time()

    from audio_sync import compute_sync_offset, export_synced_frames, save_sync_result

    # sync_verify를 output/{name}/sync_verify 에 저장
    sync_dir = os.path.join(args.output, "sync_verify")
    result = compute_sync_offset(
        args.left, args.right,
        fps=args.fps,
        out_dir=sync_dir,
        verbose=True,
    )
    save_sync_result(result, os.path.join(sync_dir, "sync_offset.txt"))

    dst = args.frames_dir
    export_synced_frames(result, left_mov=args.left, right_mov=args.right,
                         dst_dir=dst, max_frames=args.max_frames)

    # 결과 요약
    left_count = len([f for f in os.listdir(os.path.join(dst, "left")) if f.endswith(".jpg")])
    right_count = len([f for f in os.listdir(os.path.join(dst, "right")) if f.endswith(".jpg")])

    print(f"\n      Offset: {result['offset']} frames | Confidence: {result['confidence']:.1f}")
    print(f"      Exported: L={left_count}, R={right_count} frames")
    print(f"      Time: {fmt_time(time.time() - t0)}")
    return result


def step_calibrate(args):
    """[2/4] Auto Calibration"""
    print(f"\n[2/3] Auto Calibration (DISK + LightGlue)")
    t0 = time.time()

    left_frame = os.path.join(args.frames_dir, "left", "frame_000001.jpg")
    right_frame = os.path.join(args.frames_dir, "right", "frame_000001.jpg")

    if not os.path.exists(left_frame) or not os.path.exists(right_frame):
        print(f"  ERROR: Synced frames not found at {args.frames_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.calib), exist_ok=True)
    cmd = [
        sys.executable, os.path.join(BASE_DIR, "auto_calibrate.py"),
        "--left", left_frame,
        "--right", right_frame,
        "--output", args.calib,
    ]
    if args.left_focal is not None:
        cmd += ["--left-focal", str(args.left_focal)]
    if args.right_focal is not None:
        cmd += ["--right-focal", str(args.right_focal)]

    run_step("Calibration", cmd)
    print(f"      Output: {args.calib}")
    print(f"      Time: {fmt_time(time.time() - t0)}")


def step_stitch(args):
    """[3/4] Panorama Stitching"""
    print(f"\n[3/3] Panorama Stitching ({args.method})")
    t0 = time.time()

    cmd = [
        sys.executable, os.path.join(BASE_DIR, "stitch_video.py"),
        "--left", os.path.join(args.frames_dir, "left"),
        "--right", os.path.join(args.frames_dir, "right"),
        "--calib", args.calib,
        "--output", args.output,
        "--method", args.method,
    ]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    if args.fps:
        cmd += ["--fps", str(int(args.fps))]

    run_step("Stitching", cmd)
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
