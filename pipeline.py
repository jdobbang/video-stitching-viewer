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

    sync_dir = os.path.join(args.frames_dir, "sync_verify")
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
    parser.add_argument("--left", default=os.path.join(BASE_DIR, "좌측캠.MOV"),
                        help="Left camera MOV file")
    parser.add_argument("--right", default=os.path.join(BASE_DIR, "우측캠.MOV"),
                        help="Right camera MOV file")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "output"),
                        help="Output directory (default: output/)")
    parser.add_argument("--method", default="cylindrical", choices=["cylindrical", "planar"],
                        help="Stitching method (default: cylindrical)")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS override (auto-detect if omitted)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Stitching parallel workers")
    parser.add_argument("--frames-dir", default=os.path.join(BASE_DIR, "frames_sync_auto"),
                        help="Synced frames directory (default: frames_sync_auto/)")
    parser.add_argument("--calib", default=os.path.join(BASE_DIR, "calibration_auto.json"),
                        help="Calibration JSON path")

    parser.add_argument("--left-focal", type=float, default=None,
                        help="좌측 카메라 35mm 환산 focal length (mm)")
    parser.add_argument("--right-focal", type=float, default=None,
                        help="우측 카메라 35mm 환산 focal length (mm)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to extract (default: all)")

    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip audio sync (use existing frames)")
    parser.add_argument("--skip-calib", action="store_true",
                        help="Skip calibration (use existing JSON)")
    parser.add_argument("--skip-stitch", action="store_true",
                        help="Skip stitching")

    args = parser.parse_args()

    # focal 스펙이 주어지면 경로에 _spec 접미사 자동 추가
    if args.left_focal is not None:
        if not args.output.endswith("_spec"):
            args.output = args.output + "_spec"
        if not args.calib.endswith("_spec.json"):
            args.calib = args.calib.replace(".json", "_spec.json")

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
