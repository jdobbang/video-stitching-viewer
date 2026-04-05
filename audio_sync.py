"""
Audio-based Video Sync Finder (v2 - Bug Fixed)
버그 수정:
- 윈도우 오프셋 부호 수정
- 다중 윈도우 lag 계산 수정
"""

import subprocess
import wave
import os
import argparse
import numpy as np
from scipy import signal
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_ffmpeg():
    import shutil
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_fps(mov_path):
    import shutil, re
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "csv=p=0", mov_path],
                capture_output=True, text=True,
            )
            num, den = r.stdout.strip().split("/")
            return float(num) / float(den)
        except Exception:
            pass
    ffmpeg = get_ffmpeg()
    try:
        r = subprocess.run([ffmpeg, "-i", mov_path], capture_output=True, text=True)
        m = re.search(r'(\d+(?:\.\d+)?)\s*fps', r.stderr)
        if m:
            return float(m.group(1))
        m = re.search(r'(\d+)/(\d+)\s*tbr', r.stderr)
        if m:
            return float(m.group(1)) / float(m.group(2))
    except Exception:
        pass
    return None


def extract_audio(mov_path, wav_path, sample_rate=48000, max_duration=300):
    ffmpeg = get_ffmpeg()
    if os.path.exists(wav_path):
        os.remove(wav_path)
    cmd = [ffmpeg, "-i", mov_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", str(sample_rate), "-ac", "1"]
    if max_duration is not None:
        cmd += ["-t", str(max_duration)]
    cmd += [wav_path, "-y"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(wav_path):
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    with wave.open(wav_path, "rb") as w:
        dur = w.getnframes() / w.getframerate()
    print(f"  Extracted: {os.path.basename(wav_path)} ({dur:.2f}s)")
    return wav_path


def load_wav_mono(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return data, rate


# ══════════════════════════════════════════════════════════════════════════════
# 정밀 분석 함수들
# ══════════════════════════════════════════════════════════════════════════════

def refine_peak_parabolic(corr, peak_idx):
    """서브샘플 정밀도를 위한 포물선 보간"""
    if peak_idx <= 0 or peak_idx >= len(corr) - 1:
        return float(peak_idx)
    y0 = np.abs(corr[peak_idx - 1])
    y1 = np.abs(corr[peak_idx])
    y2 = np.abs(corr[peak_idx + 1])
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-10:
        return float(peak_idx)
    offset = 0.5 * (y0 - y2) / denom
    return peak_idx + offset


def compute_peak_confidence(corr, peak_idx, window=500):
    """피크 신뢰도 계산"""
    peak_val = np.abs(corr[peak_idx])
    lo = max(0, peak_idx - window)
    hi = min(len(corr), peak_idx + window)
    surrounding = np.concatenate([
        np.abs(corr[lo:max(lo, peak_idx - 10)]),
        np.abs(corr[min(hi, peak_idx + 10):hi])
    ])
    if len(surrounding) == 0:
        return 1.0
    background = np.mean(surrounding)
    if background < 1e-10:
        return float('inf')
    return peak_val / background


def multi_window_correlation(left, right, rate, num_windows=5, window_sec=3.0):
    """
    다중 윈도우 검증 (버그 수정됨)
    - 각 윈도우에서 독립적으로 lag 계산
    - 단, coarse lag를 모르므로 전체 범위에서 탐색
    """
    results = []
    window_samples = int(window_sec * rate)
    max_len = min(len(left), len(right))
    
    if max_len < window_samples * 2:
        return []
    
    step = (max_len - window_samples) // num_windows
    
    for i in range(num_windows):
        start = i * step
        end = start + window_samples
        if end > max_len:
            break
        
        left_win = left[start:end]
        right_win = right[start:end]
        
        corr = signal.correlate(left_win, right_win, mode="full", method="fft")
        lags = signal.correlation_lags(len(left_win), len(right_win), mode="full")
        
        peak_idx = np.argmax(np.abs(corr))
        refined_idx = refine_peak_parabolic(corr, peak_idx)
        
        # [버그 수정] lag 계산: lags 배열의 인덱스로 실제 lag 값 구하기
        # lags는 등차수열이므로 보간 가능
        int_idx = int(refined_idx)
        frac = refined_idx - int_idx
        if int_idx < len(lags) - 1:
            lag_samples = lags[int_idx] + frac * (lags[int_idx + 1] - lags[int_idx])
        else:
            lag_samples = lags[int_idx]
        
        lag_sec = lag_samples / rate
        confidence = compute_peak_confidence(corr, peak_idx)
        
        results.append({
            "window": i + 1,
            "start_sec": start / rate,
            "lag_sec": lag_sec,
            "lag_samples": lag_samples,
            "confidence": confidence,
        })
    
    return results


def analyze_window_consistency(window_results):
    """다중 윈도우 결과의 일관성 분석"""
    if not window_results:
        return None
    lags = [r["lag_sec"] for r in window_results]
    confidences = [r["confidence"] for r in window_results]
    return {
        "median_lag": np.median(lags),
        "mean_lag": np.mean(lags),
        "std_lag": np.std(lags),
        "min_lag": np.min(lags),
        "max_lag": np.max(lags),
        "range_ms": (np.max(lags) - np.min(lags)) * 1000,
        "mean_confidence": np.mean(confidences),
        "min_confidence": np.min(confidences),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 메인 싱크 계산 (버그 수정)
# ══════════════════════════════════════════════════════════════════════════════

def compute_sync_offset(left_mov, right_mov, fps=None, out_dir=None,
                        num_windows=5, verbose=True, precise=False):
    if out_dir is None:
        out_dir = os.path.join(BASE_DIR, "sync_verify")
    os.makedirs(out_dir, exist_ok=True)

    # FPS 자동 감지
    if fps is None:
        fps = get_fps(left_mov)
        if fps is None or fps < 1:
            cap = cv2.VideoCapture(left_mov)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        if fps is None or fps < 1:
            raise RuntimeError("FPS auto-detection failed. Use --fps.")
        if verbose:
            print(f"Detected FPS: {fps:.4f}")

    # 오디오 추출
    if verbose:
        print("Extracting audio from MOV files...")
    left_wav = extract_audio(left_mov, os.path.join(out_dir, "left_audio.wav"))
    right_wav = extract_audio(right_mov, os.path.join(out_dir, "right_audio.wav"))

    left, rate_l = load_wav_mono(left_wav)
    right, rate_r = load_wav_mono(right_wav)
    if verbose:
        print(f"Left:  {len(left)} samples, {rate_l} Hz, {len(left)/rate_l:.2f}s")
        print(f"Right: {len(right)} samples, {rate_r} Hz, {len(right)/rate_r:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # 다중 윈도우 사전 검증
    # ──────────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[다중 윈도우 검증] {num_windows}개 구간 독립 분석...")
    
    window_results = multi_window_correlation(left, right, rate_l, num_windows=num_windows)
    window_analysis = analyze_window_consistency(window_results)
    
    if window_analysis and verbose:
        print(f"  윈도우별 lag 범위: {window_analysis['range_ms']:.2f}ms")
        print(f"  표준편차: {window_analysis['std_lag']*1000:.3f}ms")
        print(f"  중앙값 lag: {window_analysis['median_lag']*1000:+.2f}ms")
        if window_analysis['range_ms'] > 50:
            print(f"  ⚠️  경고: 윈도우 간 편차가 큼 - 오디오 품질 확인 필요")

    # ──────────────────────────────────────────────────────────────────────────
    # 1단계: 대략적 위치 탐색 (precise 모드면 전체 샘플레이트 사용)
    # ──────────────────────────────────────────────────────────────────────────
    if precise:
        # 전체 샘플레이트로 직접 계산 (느리지만 정확)
        if verbose:
            print(f"\n[정밀 모드] 전체 샘플레이트({rate_l} Hz)로 계산 중... (시간 소요)")
        
        corr_full = signal.correlate(left, right, mode="full", method="fft")
        lags_full = signal.correlation_lags(len(left), len(right), mode="full")
        
        peak_idx = np.argmax(np.abs(corr_full))
        refined_idx = refine_peak_parabolic(corr_full, peak_idx)
        
        # 보간된 lag
        int_idx = int(refined_idx)
        frac = refined_idx - int_idx
        if int_idx < len(lags_full) - 1:
            total_lag_samples = lags_full[int_idx] + frac * (lags_full[int_idx + 1] - lags_full[int_idx])
        else:
            total_lag_samples = lags_full[int_idx]
        
        lag_seconds = total_lag_samples / rate_l
        lag_frames = lag_seconds * fps
        confidence = compute_peak_confidence(corr_full, peak_idx)
        
        if verbose:
            print(f"  Direct lag: {lag_seconds:+.6f}s ({lag_frames:+.3f} frames)")
            print(f"  신뢰도 점수: {confidence:.2f} ", end="")
            if confidence > 10:
                print("(✓ 매우 좋음)")
            elif confidence > 5:
                print("(✓ 좋음)")
            elif confidence > 2:
                print("(△ 보통)")
            else:
                print("(✗ 낮음)")
        
        # 그래프용 데이터 (다운샘플해서 저장)
        ds_plot = 6
        lags_coarse = lags_full[::ds_plot]
        corr_coarse = corr_full[::ds_plot]
        eff_rate = rate_l / ds_plot
        
    else:
        # 기존 2단계 방식
        ds = rate_l // 8000
        left_ds = left[::ds]
        right_ds = right[::ds]
        eff_rate = rate_l / ds
        if verbose:
            print(f"\n[1단계] Downsampled {ds}x -> {eff_rate:.0f} Hz")

        corr_coarse = signal.correlate(left_ds, right_ds, mode="full", method="fft")
        lags_coarse = signal.correlation_lags(len(left_ds), len(right_ds), mode="full")
        coarse_peak_idx = np.argmax(np.abs(corr_coarse))
        coarse_peak = lags_coarse[coarse_peak_idx]
        coarse_lag_sec = coarse_peak / eff_rate
        if verbose:
            print(f"  Coarse lag: {coarse_lag_sec:+.4f}s")

        # ──────────────────────────────────────────────────────────────────────────
        # 2단계: 원본 샘플레이트로 피크 주변 정밀 탐색
        # ──────────────────────────────────────────────────────────────────────────
        search_radius = int(rate_l * 0.5)  # ±0.5초
        coarse_lag_orig = coarse_peak * ds  # 원본 샘플 단위

        win_len = min(len(left), len(right), rate_l * 10)
        
        l_offset = max(0, coarse_lag_orig)
        r_offset = max(0, -coarse_lag_orig)
        
        l_offset = min(l_offset, len(left) - win_len)
        r_offset = min(r_offset, len(right) - win_len)
        l_offset = max(0, l_offset)
        r_offset = max(0, r_offset)
        
        left_win = left[l_offset:l_offset + win_len]
        right_win = right[r_offset:r_offset + win_len]

        if verbose:
            print(f"[2단계] Full-rate refinement")
            print(f"  Window: L[{l_offset}:+{win_len}], R[{r_offset}:+{win_len}]")
        
        corr_fine = signal.correlate(left_win, right_win, mode="full", method="fft")
        lags_fine = signal.correlation_lags(len(left_win), len(right_win), mode="full")

        center = len(lags_fine) // 2
        search_mask = np.zeros(len(corr_fine), dtype=bool)
        lo = max(0, center - search_radius)
        hi = min(len(corr_fine), center + search_radius)
        search_mask[lo:hi] = True
        corr_masked = np.where(search_mask, np.abs(corr_fine), 0)
        fine_peak_idx = np.argmax(corr_masked)

        refined_peak_idx = refine_peak_parabolic(corr_fine, fine_peak_idx)
        
        int_idx = int(refined_peak_idx)
        frac = refined_peak_idx - int_idx
        if int_idx < len(lags_fine) - 1:
            fine_lag_samples = lags_fine[int_idx] + frac * (lags_fine[int_idx + 1] - lags_fine[int_idx])
        else:
            fine_lag_samples = lags_fine[int_idx]

        total_lag_samples = fine_lag_samples + (l_offset - r_offset)
        
        lag_seconds = total_lag_samples / rate_l
        lag_frames = lag_seconds * fps
        confidence = compute_peak_confidence(corr_fine, fine_peak_idx)

        if verbose:
            print(f"  Fine lag in window: {fine_lag_samples/rate_l:+.6f}s")
            print(f"  Offset correction: {(l_offset - r_offset)/rate_l:+.6f}s")
            print(f"  Total lag: {lag_seconds:+.6f}s ({lag_frames:+.3f} frames)")
            print(f"  신뢰도 점수: {confidence:.2f} ", end="")
            if confidence > 10:
                print("(✓ 매우 좋음)")
            elif confidence > 5:
                print("(✓ 좋음)")
            elif confidence > 2:
                print("(△ 보통)")
            else:
                print("(✗ 낮음)")

    # 프레임 오프셋 계산
    offset_frames_exact = lag_frames
    offset_frames = round(lag_frames)
    subframe_offset = lag_frames - offset_frames

    # l_start, r_start 계산
    if lag_seconds > 0:
        # left가 앞서 있음 → left 시작점을 뒤로
        l_start, r_start = abs(offset_frames) + 1, 1
    else:
        # right가 앞서 있음 → right 시작점을 뒤로
        l_start, r_start = 1, abs(offset_frames) + 1

    offset = (r_start - 1) - (l_start - 1)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Peak lag: {lag_seconds:+.6f} s ({lag_seconds*1000:+.3f} ms)")
        print(f"  Frame offset @ {fps:.4f} FPS: {lag_frames:+.3f} frames")
        print(f"    → 정수 프레임: {offset_frames:+d}")
        print(f"    → 서브프레임 잔차: {subframe_offset:+.3f} ({subframe_offset/fps*1000:+.2f}ms)")
        print(f"  left_start_frame  = {l_start}")
        print(f"  right_start_frame = {r_start}")
        print(f"  offset (right - left) = {offset}")
        print(f"  신뢰도: {confidence:.2f}")
        print(f"{'='*60}")

    # 그래프 저장
    _save_correlation_plot(lags_coarse, corr_coarse, eff_rate, lag_seconds, 
                           lag_frames, fps, confidence, window_results, out_dir)

    return {
        "lag_seconds": lag_seconds,
        "lag_frames": lag_frames,
        "lag_frames_exact": offset_frames_exact,
        "subframe_offset": subframe_offset,
        "left_start": l_start,
        "right_start": r_start,
        "offset": offset,
        "fps": fps,
        "confidence": confidence,
        "window_results": window_results,
        "window_analysis": window_analysis,
    }


def _save_correlation_plot(lags, corr, eff_rate, lag_seconds, lag_frames,
                           fps, confidence, window_results, out_dir):
    lag_time = lags / eff_rate
    corr_norm = corr / np.max(np.abs(corr))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"Audio Cross-Correlation Sync Analysis (Confidence: {confidence:.1f})",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(lag_time, corr_norm, color="#4a90d9", linewidth=0.5, alpha=0.8)
    ax.axvline(lag_seconds, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Peak: {lag_seconds:+.4f}s ({lag_frames:+.2f} frames)")
    ax.axvline(0, color="#888", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Normalized Correlation")
    ax.set_title("Full Cross-Correlation")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    window = 2.0
    mask = (lag_time >= lag_seconds - window) & (lag_time <= lag_seconds + window)
    if np.any(mask):
        ax2.plot(lag_time[mask], corr_norm[mask], color="#4a90d9", linewidth=1.0)
        ax2.axvline(lag_seconds, color="#e74c3c", linewidth=2, linestyle="--")
    ax2.set_xlabel("Lag (seconds)")
    ax2.set_ylabel("Correlation")
    ax2.set_title(f"Peak Detail (±{window:.0f}s)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    if window_results:
        windows = [r["window"] for r in window_results]
        window_lags = [r["lag_sec"] * 1000 for r in window_results]
        colors = ["#2ecc71" if r["confidence"] > 5 else "#f39c12"
                  for r in window_results]
        ax3.bar(windows, window_lags, color=colors, alpha=0.7, edgecolor="black")
        ax3.axhline(lag_seconds * 1000, color="#e74c3c", linewidth=2,
                    linestyle="--", label=f"Final: {lag_seconds*1000:.2f}ms")
        ax3.set_xlabel("Window #")
        ax3.set_ylabel("Lag (ms)")
        ax3.set_title("Multi-Window Verification (green=high confidence)")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Multi-window data not available",
                 ha="center", va="center", transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "correlation_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved correlation plot: {plot_path}")


def save_sync_result(result, path):
    lines = [
        "# Sync result (audio cross-correlation) - v2 FIXED",
        f"left_start_frame  = {result['left_start']}",
        f"right_start_frame = {result['right_start']}",
        f"offset (right - left) = {result['offset']}",
        "",
        "# 정밀 분석 결과",
        f"lag_seconds = {result['lag_seconds']:+.6f}",
        f"lag_frames_exact = {result['lag_frames_exact']:+.3f}",
        f"subframe_offset = {result['subframe_offset']:+.3f}",
        f"fps = {result['fps']:.4f}",
        f"confidence = {result['confidence']:.2f}",
        "",
    ]
    if result.get("window_analysis"):
        wa = result["window_analysis"]
        lines.extend([
            "# 다중 윈도우 검증",
            f"window_median_lag = {wa['median_lag']:+.6f}",
            f"window_std_lag = {wa['std_lag']:.6f}",
            f"window_range_ms = {wa['range_ms']:.2f}",
            f"window_mean_confidence = {wa['mean_confidence']:.2f}",
        ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def get_frame_count(mov_path):
    import shutil
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        # 메타데이터 기반 (즉시 반환)
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames",
                 "-of", "csv=p=0", mov_path],
                capture_output=True, text=True,
            )
            val = r.stdout.strip()
            if val and val != "N/A":
                return int(val)
        except Exception:
            pass
        # duration * fps로 추정
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=duration,r_frame_rate",
                 "-of", "csv=p=0", mov_path],
                capture_output=True, text=True,
            )
            parts = r.stdout.strip().split(",")
            num, den = parts[0].split("/")
            fps = float(num) / float(den)
            dur = float(parts[1])
            return int(dur * fps)
        except Exception:
            pass
    cap = cv2.VideoCapture(mov_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def save_verification_frames(left_mov, right_mov, result, out_dir=None,
                             num_samples=5):
    if out_dir is None:
        out_dir = os.path.join(BASE_DIR, "sync_verify")
    os.makedirs(out_dir, exist_ok=True)

    total_l = get_frame_count(left_mov)
    total_r = get_frame_count(right_mov)

    cap_l = cv2.VideoCapture(left_mov)
    cap_r = cv2.VideoCapture(right_mov)

    l_start = result["left_start"] - 1
    r_start = result["right_start"] - 1

    overlap = min(total_l - l_start, total_r - r_start)
    if overlap <= 0:
        print("ERROR: No overlapping frames.")
        cap_l.release()
        cap_r.release()
        return out_dir

    l_end = l_start + overlap - 1
    r_end = r_start + overlap - 1

    print(f"\nSync overlap: {overlap} frames")
    print(f"  Left  : #{l_start} ~ #{l_end}  (total {total_l})")
    print(f"  Right : #{r_start} ~ #{r_end}  (total {total_r})")

    confidence = result.get("confidence", 0)
    if confidence > 10:
        conf_color = (80, 200, 80)
        conf_text = "HIGH"
    elif confidence > 5:
        conf_color = (80, 200, 255)
        conf_text = "GOOD"
    else:
        conf_color = (80, 80, 255)
        conf_text = "LOW"

    font = cv2.FONT_HERSHEY_SIMPLEX

    def add_label(img, text, color=(255, 255, 255)):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        cv2.putText(img, text, (12, 40), font, 1.0, color, 2, cv2.LINE_AA)
        return img

    sample_positions = []
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0
        frame_offset = int(t * (overlap - 1))
        lf = min(l_start + frame_offset, l_end)
        rf = min(r_start + frame_offset, r_end)
        pct = int(t * 100)
        sample_positions.append((f"{pct}%", lf, rf))

    pair_images = []
    for tag, lf, rf in sample_positions:
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, lf)
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, rf)
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            print(f"  SKIP {tag}: failed to read frame")
            continue

        h_l, w_l = frame_l.shape[:2]
        h_r, w_r = frame_r.shape[:2]
        target_h = min(h_l, h_r)
        if h_l != target_h:
            scale = target_h / h_l
            frame_l = cv2.resize(frame_l, (int(w_l * scale), target_h))
        if h_r != target_h:
            scale = target_h / h_r
            frame_r = cv2.resize(frame_r, (int(w_r * scale), target_h))

        frame_l = add_label(frame_l, f"LEFT #{lf} ({tag})", (100, 200, 255))
        frame_r = add_label(frame_r, f"RIGHT #{rf} ({tag}) [Conf:{conf_text}]", conf_color)

        pair = np.hstack([frame_l, frame_r])
        pair_images.append(pair)

        cv2.imwrite(
            os.path.join(out_dir, f"pair_{tag}_{lf:06d}_{rf:06d}.jpg"),
            pair, [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        print(f"  Saved pair {tag}: L#{lf} / R#{rf}")

    if pair_images:
        max_w = max(img.shape[1] for img in pair_images)
        resized = []
        for img in pair_images:
            h, w = img.shape[:2]
            if w != max_w:
                scale = max_w / w
                img = cv2.resize(img, (max_w, int(h * scale)))
            resized.append(img)

        grid = np.vstack(resized)
        grid_path = os.path.join(out_dir, "sync_verify_grid.jpg")
        cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved grid: {grid_path}")

    cap_l.release()
    cap_r.release()
    return out_dir


def export_synced_frames(result, left_mov=None, right_mov=None, dst_dir=None, max_frames=None):
    """ffmpeg의 select 필터로 싱크 구간 프레임만 원본 MOV에서 직접 추출

    프레임 번호는 frame_000001.jpg 부터 순차 저장
    """
    if left_mov is None:
        left_mov = os.path.join(BASE_DIR, "좌측캠.MOV")
    if right_mov is None:
        right_mov = os.path.join(BASE_DIR, "우측캠.MOV")
    if dst_dir is None:
        dst_dir = os.path.join(BASE_DIR, "frames_sync_auto")

    ffmpeg = get_ffmpeg()
    fps = result["fps"]

    l_start = result["left_start"] - 1   # 0-based
    r_start = result["right_start"] - 1

    # 총 프레임 수로 겹침 구간 계산
    total_l = get_frame_count(left_mov)
    total_r = get_frame_count(right_mov)
    overlap = min(total_l - l_start, total_r - r_start)

    if overlap <= 0:
        print("ERROR: No overlapping frames to export.")
        return

    if max_frames is not None:
        overlap = min(overlap, max_frames)

    l_end = l_start + overlap - 1
    r_end = r_start + overlap - 1

    print(f"\nExporting synced frames via ffmpeg...")
    print(f"  Left  : frame {l_start} ~ {l_end}  (of {total_l})")
    print(f"  Right : frame {r_start} ~ {r_end}  (of {total_r})")
    print(f"  Overlap: {overlap} frames")

    jobs = [
        ("left",  left_mov,  l_start, overlap),
        ("right", right_mov, r_start, overlap),
    ]

    for tag, mov, start, num_frames in jobs:
        out_path = os.path.join(dst_dir, tag)
        os.makedirs(out_path, exist_ok=True)

        # -ss로 시작 지점까지 빠르게 seek, -frames:v로 필요한 프레임 수만 추출
        ss_sec = start / fps
        cmd = [
            ffmpeg,
            "-ss", f"{ss_sec:.6f}",
            "-i", mov,
            "-frames:v", str(num_frames),
            "-qscale:v", "2",
            "-start_number", "1",
            os.path.join(out_path, "frame_%06d.jpg"),
            "-y",
        ]

        print(f"  Extracting {tag} (seek to {ss_sec:.2f}s, {num_frames} frames)...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  ERROR ({tag}): {proc.stderr[:300]}")
        else:
            count = len([f for f in os.listdir(out_path) if f.endswith(".jpg")])
            print(f"  {tag}: {count} frames saved")

    print(f"  Done! Saved to: {dst_dir}")
    return dst_dir


def iter_synced_frames(result, left_mov, right_mov, max_frames=None):
    """싱크된 프레임 쌍을 메모리에서 직접 yield하는 제너레이터.

    디스크 JPEG I/O 없이 cv2.VideoCapture로 직접 디코딩.
    yields (left_frame, right_frame) BGR numpy 배열 쌍.
    """
    fps = result["fps"]
    l_start = result["left_start"] - 1  # 0-based
    r_start = result["right_start"] - 1

    total_l = get_frame_count(left_mov)
    total_r = get_frame_count(right_mov)
    overlap = min(total_l - l_start, total_r - r_start)

    if overlap <= 0:
        return

    if max_frames is not None:
        overlap = min(overlap, max_frames)

    cap_l = cv2.VideoCapture(left_mov)
    cap_r = cv2.VideoCapture(right_mov)

    # 시작 프레임으로 seek
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, l_start)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, r_start)

    try:
        for _ in range(overlap):
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                break
            yield frame_l, frame_r
    finally:
        cap_l.release()
        cap_r.release()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio-based sync finder v2 (bug fixed)"
    )
    parser.add_argument("--left", default=os.path.join(BASE_DIR, "좌측캠.MOV"))
    parser.add_argument("--right", default=os.path.join(BASE_DIR, "우측캠.MOV"))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--precise", action="store_true",
                        help="전체 샘플레이트로 계산 (느리지만 정확)")
    parser.add_argument("--no-verify", action="store_true",
                        help="시각화 검증 프레임 저장 건너뛰기")
    parser.add_argument("--export", action="store_true",
                        help="싱크 맞춘 프레임을 ffmpeg로 frames_sync_auto/에 추출")
    parser.add_argument("--dst-frames", default=None,
                        help="싱크 프레임 출력 폴더 (default: frames_sync_auto/)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(BASE_DIR, "sync_verify")

    result = compute_sync_offset(
        args.left, args.right,
        fps=args.fps,
        out_dir=out_dir,
        num_windows=args.windows,
        precise=args.precise
    )

    save_sync_result(result, os.path.join(out_dir, "sync_offset.txt"))

    if not args.no_verify:
        save_verification_frames(args.left, args.right, result,
                                 out_dir=out_dir, num_samples=args.samples)

    if args.export:
        export_synced_frames(result,
                             left_mov=args.left,
                             right_mov=args.right,
                             dst_dir=args.dst_frames)

    print(f"\n{'='*60}")
    print("📊 분석 완료")
    print(f"{'='*60}")
    print(f"  정밀 오프셋: {result['lag_frames_exact']:+.3f} frames")
    print(f"  서브프레임 잔차: {result['subframe_offset']:+.3f} frames")
    print(f"  신뢰도: {result['confidence']:.1f}", end=" ")
    if result['confidence'] > 10:
        print("✓✓")
    elif result['confidence'] > 5:
        print("✓")
    elif result['confidence'] > 2:
        print("△")
    else:
        print("✗")

    if result.get("window_analysis"):
        wa = result["window_analysis"]
        print(f"  윈도우 일관성: 범위 {wa['range_ms']:.2f}ms, 표준편차 {wa['std_lag']*1000:.2f}ms")

    print(f"\n  저장 위치: {out_dir}")
    print(f"{'='*60}")