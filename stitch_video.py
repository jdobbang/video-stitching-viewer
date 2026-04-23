#!/usr/bin/env python3
"""
Panorama Video Stitching Pipeline
==================================
동기화된 좌/우 카메라 프레임을 파노라마 영상으로 스티칭한다.

스티칭 흐름:
    1. calibration.json에서 대응점 로드
    2. focal_weight로 focal_length 결정 (auto: 좌/우 크기 균형 기준)
    3. cylindrical 투영 → 호모그래피 계산 → 블렌딩 마스크 사전 계산
    4. 매 프레임: cylindrical warp → 호모그래피 적용 → 블렌딩 합성

Usage:
    python stitch_video.py --left ./left_sync --right ./right_sync --calib calibration.json
    python stitch_video.py --left ... --right ... --calib ... --focal-weight 0.8
    python stitch_video.py --video-only --frames ./output/frames --fps 30

Requirements:
    pip install opencv-python numpy tqdm
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def _check_cuda():
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return True
    except (AttributeError, cv2.error):
        pass
    return False


HAS_CUDA = _check_cuda()


# ============================================================================
# CYLINDRICAL PROJECTION
# ============================================================================

def build_cylindrical_maps(h, w, focal_length):
    """Cylindrical remap 맵을 사전 계산한다 (프레임마다 재사용)."""
    cx, cy = w / 2, h / 2
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)

    theta = (x_grid - cx) / focal_length
    h_c = (y_grid - cy) / focal_length

    map_x = (focal_length * np.tan(theta) + cx).astype(np.float32)
    map_y = (h_c / np.cos(theta) * focal_length + cy).astype(np.float32)
    return map_x, map_y


def cylindrical_warp(img, focal_length, maps=None):
    """이미지에 cylindrical 투영을 적용한다."""
    if maps is None:
        h, w = img.shape[:2]
        maps = build_cylindrical_maps(h, w, focal_length)
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def planar_to_cylindrical(x, y, focal_length, cx, cy):
    """planar 좌표를 cylindrical 좌표로 변환한다."""
    theta = np.arctan((x - cx) / focal_length)
    x_cyl = focal_length * theta + cx
    y_cyl = (y - cy) * np.cos(theta) + cy
    return x_cyl, y_cyl


def get_valid_bounds(img):
    """검은색이 아닌 픽셀의 바운딩박스를 구한다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.where(gray > 0)
    if len(coords[0]) == 0:
        return 0, img.shape[0] - 1, 0, img.shape[1] - 1
    return coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()


# ============================================================================
# FOCAL WEIGHT AUTO
# ============================================================================

def find_optimal_focal_weight(point_pairs, w, h, step=0.05):
    """좌/우 프레임 크기가 가장 비슷해지는 focal_weight를 탐색한다.

    각 weight에서 cylindrical 호모그래피를 계산하고,
    좌측(원본 크기)과 우측(H 변환 후 크기)의 너비 비율이
    1.0에 가장 가까운 weight를 반환한다.
    """
    cx, cy = w / 2, h / 2
    best = 0.5
    best_ratio_diff = float('inf')

    for weight in np.arange(0.3, 1.05, step):
        f = w * weight

        # 대응점을 cylindrical 좌표로 변환
        pts_l, pts_r = [], []
        for pp in point_pairs:
            lx, ly = planar_to_cylindrical(pp['left'][0], pp['left'][1], f, cx, cy)
            rx, ry = planar_to_cylindrical(pp['right'][0], pp['right'][1], f, cx, cy)
            pts_l.append([lx, ly])
            pts_r.append([rx, ry])
        pts_l = np.array(pts_l, dtype=np.float32)
        pts_r = np.array(pts_r, dtype=np.float32)

        H, _ = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 5.0)
        if H is None:
            continue

        # cylindrical 유효 영역
        dummy = np.ones((h, w, 3), dtype=np.uint8) * 255
        cyl_img = cylindrical_warp(dummy, f)
        y1, y2, x1, x2 = get_valid_bounds(cyl_img)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           dtype=np.float32).reshape(-1, 1, 2)

        # 좌측 너비 (원본 그대로)
        left_w = x2 - x1

        # 우측 너비 (H 변환 후)
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        right_w = warped[:, 0].max() - warped[:, 0].min()

        # 발산 체크
        if right_w > w * 3 or right_w < w * 0.2:
            continue

        ratio = right_w / left_w if left_w > 0 else 999
        diff = abs(ratio - 1.0)

        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best = weight

    return round(best, 2)


# ============================================================================
# STITCHER
# ============================================================================

class PanoramaStitcher:
    """좌/우 이미지 쌍을 파노라마로 스티칭한다.

    초기화 시 호모그래피, 캔버스, 블렌딩 마스크를 사전 계산하여
    매 프레임 스티칭 시 warpPerspective + 행렬 연산만 수행한다.
    """

    def __init__(self, calibration_path, method='cylindrical', focal_weight='auto',
                 lab_match=True, pixel_match=True, multi_band=True, mbb_num_bands=5):
        """
        Args:
            calibration_path: calibration.json 경로
            method: 'cylindrical' (원통 투영) 또는 'planar' (평면)
            focal_weight: focal_length = width * focal_weight
                          'auto' = 좌/우 크기 균형 기준 자동 결정
            pixel_match: True면 픽셀 이퀄라이징(BGR 게인 + Lab a/b) 수행.
                         False면 전부 스킵하고 raw warped 픽셀로 블렌딩 (기본: True).
            lab_match: pixel_match=True일 때만 의미 있음. BGR 게인 보정 후 Lab a/b
                       채널 평균을 추가로 매칭 (기본: True).
            multi_band: True면 Laplacian pyramid 기반 Multi-band blending 사용
                        (저주파는 넓게, 고주파는 좁게 블렌딩하여 시임 최소화).
                        False면 기존 선형 알파 블렌딩 (기본: True).
            mbb_num_bands: Multi-band blending pyramid 레벨 수 (기본: 5).
        """
        with open(calibration_path, 'r') as f:
            self.calib = json.load(f)

        self.method = method
        self.pixel_match = pixel_match
        self.lab_match = lab_match
        self.multi_band = multi_band
        self.mbb_num_bands = mbb_num_bands
        self.image_size = tuple(self.calib['image_size'])
        self.w, self.h = self.image_size

        # ── focal length 결정 ──
        if focal_weight == 'auto':
            if 'left_focal_mm' in self.calib:
                # 스펙 기반 초기값 → 좌/우 1:1 비율로 미세 조정
                init_weight = round(self.calib['left_focal_mm'] / 36, 2)
                focal_weight = self._refine_focal_weight(init_weight)
                print(f"Focal weight (spec): {init_weight} → {focal_weight}"
                      f" (left={self.calib['left_focal_mm']}mm, 1:1 보정)")
            else:
                focal_weight = find_optimal_focal_weight(
                    self.calib['point_pairs'], self.w, self.h)
                print(f"Focal weight (auto): {focal_weight}")
        self.focal_length = self.w * focal_weight

        # ── cylindrical remap 맵 사전 계산 ──
        self._cyl_maps = None
        if method == 'cylindrical':
            self._cyl_maps = build_cylindrical_maps(self.h, self.w, self.focal_length)

        # ── 호모그래피 / 캔버스 / 블렌딩 사전 계산 ──
        # (_precompute_blend_weight 내부에서 multi_band=True면 _precompute_mbb_pyramid 호출)
        self._mbb_alpha_pyr = None
        self._mbb_valid_mask_3c = None
        self._mbb_num_bands = 0
        self._compute_homography()
        self._compute_canvas_params()
        self._build_combined_maps()
        self._precompute_blend_weight()

        # ── 노출 보정 LUT (첫 프레임에서 계산) ──
        self._exposure_luts = None
        # ── Lab a/b 채널 시프트 (첫 프레임에서 계산, lab_match=True일 때만) ──
        self._ab_shifts = None

        # ── GPU remap 맵 업로드 ──
        self._gpu_maps = None
        if HAS_CUDA and self._left_combined_maps is not None:
            try:
                self._gpu_left_map_x = cv2.cuda_GpuMat(self._left_combined_maps[0])
                self._gpu_left_map_y = cv2.cuda_GpuMat(self._left_combined_maps[1])
                self._gpu_right_map_x = cv2.cuda_GpuMat(self._right_combined_maps[0])
                self._gpu_right_map_y = cv2.cuda_GpuMat(self._right_combined_maps[1])
                self._gpu_maps = True
                print(f"  GPU remap maps uploaded")
            except Exception as e:
                print(f"  GPU map upload failed, CPU fallback: {e}")
                self._gpu_maps = None

        print(f"Stitcher initialized: {method} method")
        print(f"  Image size: {self.w}x{self.h}")
        print(f"  Focal length: {self.focal_length:.0f}")
        print(f"  Canvas size: {self.canvas_w}x{self.canvas_h}")

    # ── focal weight 미세 조정 ─────────────────────────────────────
    def _refine_focal_weight(self, init_weight, search_range=0.3, step=0.05):
        """스펙 기반 초기 weight 주변에서 좌/우 1:1 비율이 되는 값을 탐색한다.

        init_weight를 중심으로 ±search_range 범위를 탐색하여
        좌/우 너비 비율이 1.0에 가장 가까운 weight를 반환한다.
        """
        cx, cy = self.w / 2, self.h / 2
        best = init_weight
        best_diff = float('inf')

        lo = max(0.2, init_weight - search_range)
        hi = min(1.5, init_weight + search_range)

        for weight in np.arange(lo, hi + step, step):
            f = self.w * weight
            pts_l, pts_r = [], []
            for pp in self.calib['point_pairs']:
                lx, ly = planar_to_cylindrical(
                    pp['left'][0], pp['left'][1], f, cx, cy)
                rx, ry = planar_to_cylindrical(
                    pp['right'][0], pp['right'][1], f, cx, cy)
                pts_l.append([lx, ly])
                pts_r.append([rx, ry])
            pts_l = np.array(pts_l, dtype=np.float32)
            pts_r = np.array(pts_r, dtype=np.float32)

            H, _ = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 5.0)
            if H is None:
                continue

            dummy = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
            cyl_img = cylindrical_warp(dummy, f)
            y1, y2, x1, x2 = get_valid_bounds(cyl_img)
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                               dtype=np.float32).reshape(-1, 1, 2)

            left_w = x2 - x1
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            right_w = warped[:, 0].max() - warped[:, 0].min()

            if right_w > self.w * 3 or right_w < self.w * 0.2:
                continue

            diff = abs(right_w / left_w - 1.0)
            if diff < best_diff:
                best_diff = diff
                best = weight

        return round(best, 2)

    # ── 호모그래피 계산 ──────────────────────────────────────────────
    def _compute_homography(self):
        """대응점에서 호모그래피(right -> left)를 계산한다.

        cylindrical: 대응점을 cylindrical 좌표로 변환 후 계산
        planar: calibration.json의 homography 사용
        """
        cx, cy = self.w / 2, self.h / 2

        if self.method == 'cylindrical':
            cyl_left, cyl_right = [], []
            for pp in self.calib['point_pairs']:
                lx, ly = planar_to_cylindrical(
                    pp['left'][0], pp['left'][1], self.focal_length, cx, cy)
                rx, ry = planar_to_cylindrical(
                    pp['right'][0], pp['right'][1], self.focal_length, cx, cy)
                cyl_left.append([lx, ly])
                cyl_right.append([rx, ry])
            cyl_left = np.array(cyl_left, dtype=np.float32)
            cyl_right = np.array(cyl_right, dtype=np.float32)
            self.H, _ = cv2.findHomography(cyl_right, cyl_left, cv2.RANSAC, 5.0)
        else:
            self.H = np.array(self.calib['homography'], dtype=np.float64)

    # ── 캔버스 파라미터 ──────────────────────────────────────────────
    def _compute_canvas_params(self):
        """캔버스 크기, 오프셋, 변환 행렬을 사전 계산한다."""
        dummy = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255

        if self.method == 'cylindrical':
            cyl_img = cylindrical_warp(dummy, self.focal_length)
            y1, y2, x1, x2 = get_valid_bounds(cyl_img)
        else:
            y1, y2, x1, x2 = 0, self.h - 1, 0, self.w - 1

        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        ).reshape(-1, 1, 2)

        warped_right = cv2.perspectiveTransform(corners, self.H).reshape(-1, 2)
        all_corners = np.vstack([corners.reshape(-1, 2), warped_right])

        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        self.canvas_w = x_max - x_min
        self.canvas_h = y_max - y_min
        self.offset_x = -x_min
        self.offset_y = -y_min

        self.T = np.array([
            [1, 0, self.offset_x],
            [0, 1, self.offset_y],
            [0, 0, 1]
        ], dtype=np.float64)

        self.H_translated = self.T @ self.H

    # ── 합성 remap 맵 사전 계산 ────────────────────────────────────
    def _build_combined_maps(self):
        """cylindrical warp + 캔버스 배치를 하나의 remap 맵으로 합성한다.

        remap은 dst 픽셀 → src 픽셀 역매핑이므로,
        캔버스 좌표 (canvas_h x canvas_w) → 원본 이미지 좌표 (h x w) 맵을 만든다.

        좌측: canvas(dst) → T역변환 → cylindrical 좌표 → 역투영 → 원본 픽셀
        우측: canvas(dst) → H_translated 역변환 → cylindrical 좌표 → 역투영 → 원본 픽셀
        """
        self._left_combined_maps = None
        self._right_combined_maps = None

        if self._cyl_maps is None:
            return

        cx, cy_c = self.w / 2, self.h / 2
        f = self.focal_length

        # 캔버스 좌표 그리드
        xs = np.arange(self.canvas_w, dtype=np.float64)
        ys = np.arange(self.canvas_h, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)

        # ── 좌측: canvas → T^-1 → cylindrical 좌표 → 원본 픽셀 ──
        # T는 단순 평행이동이므로 T^-1은 오프셋을 빼는 것
        cyl_x = grid_x - self.offset_x
        cyl_y = grid_y - self.offset_y

        # cylindrical 역투영: cyl 좌표 → 원본 planar 좌표
        theta = (cyl_x - cx) / f
        left_mx = (f * np.tan(theta) + cx).astype(np.float32)
        left_my = ((cyl_y - cy_c) / np.cos(theta) + cy_c).astype(np.float32)
        self._left_combined_maps = (left_mx, left_my)

        # ── 우측: canvas → H_translated^-1 → cylindrical 좌표 → 원본 픽셀 ──
        H_inv = np.linalg.inv(self.H_translated)
        denom = H_inv[2, 0] * grid_x + H_inv[2, 1] * grid_y + H_inv[2, 2]
        mid_x = (H_inv[0, 0] * grid_x + H_inv[0, 1] * grid_y + H_inv[0, 2]) / denom
        mid_y = (H_inv[1, 0] * grid_x + H_inv[1, 1] * grid_y + H_inv[1, 2]) / denom

        theta_r = (mid_x - cx) / f
        right_mx = (f * np.tan(theta_r) + cx).astype(np.float32)
        right_my = ((mid_y - cy_c) / np.cos(theta_r) + cy_c).astype(np.float32)
        self._right_combined_maps = (right_mx, right_my)

    # ── 블렌딩 마스크 사전 계산 ──────────────────────────────────────
    def _precompute_blend_weight(self):
        """좌/우 마스크와 겹침 영역 선형 블렌딩 가중치를 사전 계산한다.

        카메라 배치가 고정이므로 겹침 영역은 매 프레임 동일하다.
        """
        dummy = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255

        if self._left_combined_maps is not None:
            canvas_left = cv2.remap(
                dummy, self._left_combined_maps[0], self._left_combined_maps[1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_right = cv2.remap(
                dummy, self._right_combined_maps[0], self._right_combined_maps[1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            if self._cyl_maps is not None:
                dummy = cylindrical_warp(dummy, self.focal_length, maps=self._cyl_maps)
            warped_right = cv2.warpPerspective(
                dummy, self.H_translated, (self.canvas_w, self.canvas_h))
            canvas_left = cv2.warpPerspective(
                dummy, self.T, (self.canvas_w, self.canvas_h))

        mask_left = (canvas_left > 0).any(axis=2).astype(np.float32)
        mask_right = (warped_right > 0).any(axis=2).astype(np.float32)
        overlap = mask_left * mask_right

        # 겹침 영역 행별 선형 블렌딩
        blend_weight = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
        for y in range(self.canvas_h):
            cols = np.where(overlap[y, :] > 0)[0]
            if len(cols) > 1:
                x_start, x_end = cols[0], cols[-1]
                blend_weight[y, x_start:x_end + 1] = np.linspace(
                    0, 1, x_end - x_start + 1)

        # uint8 마스크로 캐싱 (프레임별 float32 변환 제거)
        mask_left_only = (mask_left * (1 - mask_right)).astype(bool)
        mask_right_only = (mask_right * (1 - mask_left)).astype(bool)
        overlap_mask = overlap.astype(bool)

        # 3채널 bool 마스크
        self._mask_left_only_3c = np.stack([mask_left_only] * 3, axis=2)
        self._mask_right_only_3c = np.stack([mask_right_only] * 3, axis=2)
        self._overlap_mask_3c = np.stack([overlap_mask] * 3, axis=2)

        # 블렌딩 alpha를 uint16으로 저장 (0~256 범위, 정수 나눗셈용)
        # alpha=0 → 좌측 100%, alpha=256 → 우측 100%
        self._blend_alpha = (blend_weight * 256).astype(np.uint16)
        self._blend_alpha_3c = np.stack([self._blend_alpha] * 3, axis=2)
        self._has_overlap = np.any(overlap_mask)

        # 유효 영역 bool 마스크 (crop_rect + MBB에서 사용, 1회성이라 로컬)
        any_valid = (mask_left + mask_right) > 0

        # 크롭 영역: 블렌딩 없이 유효 영역 bbox만 계산 (동일 결과, 훨씬 단순)
        coords = cv2.findNonZero(any_valid.astype(np.uint8))
        self._crop_rect = cv2.boundingRect(coords) if coords is not None else None

        # Multi-band blending 피라미드 (옵션)
        if self.multi_band and self._has_overlap:
            self._precompute_mbb_pyramid(blend_weight, mask_right_only, any_valid)

    # ── Multi-band blending 사전 계산 ───────────────────────────────
    def _precompute_mbb_pyramid(self, blend_weight, mask_right_only, any_valid):
        """Multi-band blending 용 알파 마스크 Gaussian pyramid 사전 계산.

        알파 마스크 정의 (캔버스 전체):
            - left-only / 무효 영역: 0 (좌측 100% 또는 크롭 제외)
            - overlap 영역: 기존 선형 그라디언트 0→1
            - right-only 영역: 1 (우측 100%)
        """
        # 전체 캔버스 알파 (float32, 0~1)
        alpha = blend_weight.copy()
        alpha[mask_right_only] = 1.0

        # 실제 가능한 최대 레벨 (해상도 하한 고려)
        min_dim = min(self.canvas_h, self.canvas_w)
        max_levels = max(1, int(np.log2(min_dim)) - 3)
        n = min(self.mbb_num_bands, max_levels)
        self._mbb_num_bands = n

        # Gaussian pyramid (float32, 3채널 확장 — per-frame 연산 최소화)
        pyr = [alpha]
        for _ in range(n):
            pyr.append(cv2.pyrDown(pyr[-1]))
        self._mbb_alpha_pyr = [p[..., np.newaxis].astype(np.float32) for p in pyr]

        # "어느 쪽이든 유효" 3채널 mask (최종 출력 clipping 용)
        self._mbb_valid_mask_3c = np.stack([any_valid] * 3, axis=2)

        print(f"  Multi-band blending: {n} bands, canvas={self.canvas_w}x{self.canvas_h}")

    def _multi_band_blend(self, left, right):
        """두 워프 이미지를 Laplacian pyramid 기반으로 블렌딩.

        Args:
            left, right: BGR uint8 (canvas_h, canvas_w, 3)
        Returns:
            블렌딩된 BGR uint8 (canvas_h, canvas_w, 3)
        """
        n = self._mbb_num_bands
        L = left.astype(np.float32)
        R = right.astype(np.float32)

        # Gaussian pyramids
        gL = [L]
        gR = [R]
        for _ in range(n):
            gL.append(cv2.pyrDown(gL[-1]))
            gR.append(cv2.pyrDown(gR[-1]))

        # Laplacian pyramids (top level = coarsest Gaussian)
        lapL, lapR = [], []
        for i in range(n):
            dst_size = (gL[i].shape[1], gL[i].shape[0])
            upL = cv2.pyrUp(gL[i + 1], dstsize=dst_size)
            upR = cv2.pyrUp(gR[i + 1], dstsize=dst_size)
            lapL.append(gL[i] - upL)
            lapR.append(gR[i] - upR)
        lapL.append(gL[n])
        lapR.append(gR[n])

        # 레벨별 블렌딩
        blended = []
        for i in range(n + 1):
            a = self._mbb_alpha_pyr[i]
            # 크기 mismatch 방어 (pyrDown의 반올림 차이로 1px 어긋날 수 있음)
            if a.shape[:2] != lapL[i].shape[:2]:
                a = cv2.resize(a[:, :, 0], (lapL[i].shape[1], lapL[i].shape[0]))[..., np.newaxis]
            blended.append(lapL[i] * (1.0 - a) + lapR[i] * a)

        # 재구성 (coarsest에서 시작, pyrUp + Laplacian 더하기)
        out = blended[n]
        for i in range(n - 1, -1, -1):
            dst_size = (blended[i].shape[1], blended[i].shape[0])
            out = cv2.pyrUp(out, dstsize=dst_size)
            out = out + blended[i]

        return np.clip(out, 0, 255).astype(np.uint8)

    # ── 노출 보정 ─────────────────────────────────────────────────────
    def _compute_exposure_luts(self, canvas_left, warped_right):
        """첫 프레임의 겹침 영역 평균 밝기로 노출 보정 LUT를 생성한다.

        카메라 고정이므로 gain은 전 프레임에 동일하게 적용된다.
        """
        overlap_1c = self._overlap_mask_3c[:, :, 0]
        l_pixels = canvas_left[overlap_1c].reshape(-1, 3).astype(np.float64)
        r_pixels = warped_right[overlap_1c].reshape(-1, 3).astype(np.float64)

        mean_l = l_pixels.mean(axis=0)  # BGR
        mean_r = r_pixels.mean(axis=0)
        target = (mean_l + mean_r) / 2.0

        base = np.arange(256, dtype=np.float64)

        def make_luts(mean_val):
            gains = np.where(mean_val > 1, target / mean_val, 1.0)
            return tuple(
                np.clip(base * gains[c], 0, 255).astype(np.uint8)
                for c in range(3)
            )

        self._exposure_luts = (make_luts(mean_l), make_luts(mean_r))

        gain_l = target / np.maximum(mean_l, 1)
        gain_r = target / np.maximum(mean_r, 1)
        print(f"  Exposure compensation: "
              f"L=[{gain_l[2]:.3f}, {gain_l[1]:.3f}, {gain_l[0]:.3f}] "
              f"R=[{gain_r[2]:.3f}, {gain_r[1]:.3f}, {gain_r[0]:.3f}]")

        # ── Lab a/b 채널 매칭 (옵션) ─────────────────────────────────
        # BGR 게인 보정 후 잔여 색감(white balance) 차이를 Lab a/b 평균
        # 시프트로 양쪽이 중간값에 수렴하도록 보정.
        if self.lab_match:
            lut_l, lut_r = self._exposure_luts
            l_corr = self._apply_lut(canvas_left, lut_l)
            r_corr = self._apply_lut(warped_right, lut_r)

            l_lab = cv2.cvtColor(l_corr, cv2.COLOR_BGR2LAB)
            r_lab = cv2.cvtColor(r_corr, cv2.COLOR_BGR2LAB)

            l_ab = l_lab[overlap_1c][:, 1:3].astype(np.float64).mean(axis=0)
            r_ab = r_lab[overlap_1c][:, 1:3].astype(np.float64).mean(axis=0)
            target_ab = (l_ab + r_ab) / 2.0

            shift_l = (target_ab - l_ab).astype(np.float32)
            shift_r = (target_ab - r_ab).astype(np.float32)
            self._ab_shifts = (shift_l, shift_r)

            print(f"  Lab a/b matching: "
                  f"L=[Δa={shift_l[0]:+.2f}, Δb={shift_l[1]:+.2f}] "
                  f"R=[Δa={shift_r[0]:+.2f}, Δb={shift_r[1]:+.2f}]")

    @staticmethod
    def _apply_lut(img, luts):
        """3채널 LUT를 적용한다 (cv2.LUT 기반, 매우 빠름)."""
        b, g, r = cv2.split(img)
        return cv2.merge([cv2.LUT(b, luts[0]), cv2.LUT(g, luts[1]), cv2.LUT(r, luts[2])])

    @staticmethod
    def _apply_ab_shift(img, shift):
        """Lab a/b 채널에 상수 시프트를 적용한다 (L 채널은 보존)."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + int(round(float(shift[0]))), 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + int(round(float(shift[1]))), 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── 스티칭 ───────────────────────────────────────────────────────
    def stitch(self, left_img, right_img, crop=True):
        """좌/우 이미지를 스티칭한다.

        Args:
            left_img: 좌측 카메라 이미지 (BGR)
            right_img: 우측 카메라 이미지 (BGR)
            crop: 검은 테두리 크롭 여부

        Returns:
            스티칭된 파노라마 이미지
        """
        canvas_size = (self.canvas_w, self.canvas_h)

        # 1+2) 합성 remap 맵으로 cylindrical warp + 캔버스 배치를 1회로 처리
        # NOTE: borderMode=BORDER_REPLICATE 사용 — 매핑이 원본 바깥을 가리킬 때
        # 검은 픽셀(0,0,0) 대신 가장자리 픽셀을 복제. 알파 블렌딩 시 검은 띠가
        # 노출되는 것을 방지 (특히 좌/우 해상도가 달라 캘리브가 살짝 어긋날 때).
        # 유효 영역 마스크는 _precompute_blend_weight에서 BORDER_CONSTANT로 따로 계산됨.
        if self._gpu_maps:
            # GPU 경로: 프레임 업로드 → GPU remap → 다운로드
            gpu_left = cv2.cuda_GpuMat(left_img)
            gpu_right = cv2.cuda_GpuMat(right_img)
            gpu_canvas_left = cv2.cuda.remap(
                gpu_left, self._gpu_left_map_x, self._gpu_left_map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            gpu_warped_right = cv2.cuda.remap(
                gpu_right, self._gpu_right_map_x, self._gpu_right_map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            canvas_left = gpu_canvas_left.download()
            warped_right = gpu_warped_right.download()
        elif self._left_combined_maps is not None:
            canvas_left = cv2.remap(
                left_img, self._left_combined_maps[0], self._left_combined_maps[1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            warped_right = cv2.remap(
                right_img, self._right_combined_maps[0], self._right_combined_maps[1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            # planar 모드: 기존 방식
            canvas_left = cv2.warpPerspective(
                left_img, self.T, canvas_size, borderMode=cv2.BORDER_REPLICATE)
            warped_right = cv2.warpPerspective(
                right_img, self.H_translated, canvas_size, borderMode=cv2.BORDER_REPLICATE)

        # 2.5) 픽셀 이퀄라이징 (노출 보정 LUT + Lab a/b 시프트, 첫 프레임에서 계산 후 캐싱)
        if self.pixel_match and self._has_overlap:
            if self._exposure_luts is None:
                self._compute_exposure_luts(canvas_left, warped_right)
            lut_l, lut_r = self._exposure_luts
            canvas_left = self._apply_lut(canvas_left, lut_l)
            warped_right = self._apply_lut(warped_right, lut_r)

            # 2.6) Lab a/b 색감 매칭 (옵션, 첫 프레임 시프트 캐시 사용)
            if self._ab_shifts is not None:
                shift_l, shift_r = self._ab_shifts
                canvas_left = self._apply_ab_shift(canvas_left, shift_l)
                warped_right = self._apply_ab_shift(warped_right, shift_r)

        # 3) 블렌딩
        if self.multi_band and self._has_overlap and self._mbb_alpha_pyr is not None:
            # Multi-band blending (Laplacian pyramid)
            blended = self._multi_band_blend(canvas_left, warped_right)
            # 유효 영역만 살리고 나머지는 0
            result = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
            result[self._mbb_valid_mask_3c] = blended[self._mbb_valid_mask_3c]
        else:
            # 기존 선형 알파 블렌딩 (uint8, float32 변환 없음)
            result = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

            # 비겹침 영역: 직접 복사
            result[self._mask_left_only_3c] = canvas_left[self._mask_left_only_3c]
            result[self._mask_right_only_3c] = warped_right[self._mask_right_only_3c]

            # 겹침 영역: uint16 정수 블렌딩 (float 없이)
            if self._has_overlap:
                alpha = self._blend_alpha_3c[self._overlap_mask_3c].astype(np.uint16)
                l_vals = canvas_left[self._overlap_mask_3c].astype(np.uint16)
                r_vals = warped_right[self._overlap_mask_3c].astype(np.uint16)
                result[self._overlap_mask_3c] = (
                    (l_vals * (256 - alpha) + r_vals * alpha) >> 8
                ).astype(np.uint8)

        # 4) 크롭
        if crop and self._crop_rect is not None:
            x, y, rw, rh = self._crop_rect
            result = result[y:y + rh, x:x + rw]

        return result


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_single_frame(args):
    """단일 프레임 스티칭 (멀티프로세싱용)."""
    idx, left_path, right_path, stitcher_params, output_path = args

    stitcher = PanoramaStitcher.__new__(PanoramaStitcher)
    stitcher.__dict__.update(stitcher_params)

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        return idx, False, "Failed to load images"

    try:
        result = stitcher.stitch(left_img, right_img, crop=True)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


def batch_stitch(left_dir, right_dir, calibration_path, output_dir,
                 method='cylindrical', num_workers=None,
                 frame_pattern='frame_*.jpg', focal_weight='auto',
                 lab_match=True, pixel_match=True, multi_band=True):
    """전체 프레임을 일괄 스티칭한다.

    Args:
        left_dir: 좌측 프레임 디렉토리
        right_dir: 우측 프레임 디렉토리
        calibration_path: calibration.json 경로
        output_dir: 출력 디렉토리
        method: 'cylindrical' 또는 'planar'
        num_workers: 병렬 워커 수 (기본: CPU - 1)
        frame_pattern: 프레임 파일 패턴
        focal_weight: focal_length = width * focal_weight ('auto' 또는 숫자)
    """
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 프레임 탐색
    left_frames = sorted(left_dir.glob(frame_pattern))
    if not left_frames:
        for pattern in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
            left_frames = sorted(left_dir.glob(pattern))
            if left_frames:
                break

    print(f"Found {len(left_frames)} frames in left directory")

    # 좌/우 프레임 매칭
    frame_pairs = []
    for left_path in left_frames:
        right_path = right_dir / left_path.name
        if not right_path.exists():
            alt_name = left_path.name.replace('left', 'right').replace('LEFT', 'RIGHT')
            right_path = right_dir / alt_name
        if right_path.exists():
            output_path = output_dir / f"stitched_{left_path.stem}.jpg"
            frame_pairs.append((left_path, right_path, output_path))

    print(f"Matched {len(frame_pairs)} frame pairs")
    if not frame_pairs:
        print("No frame pairs found!")
        return

    # Stitcher 초기화
    stitcher = PanoramaStitcher(calibration_path, method=method,
                                focal_weight=focal_weight,
                                lab_match=lab_match, pixel_match=pixel_match,
                                multi_band=multi_band)

    # 멀티프로세싱용 파라미터 직렬화 (GPU 관련 필드 제외)
    stitcher_params = {
        'method': stitcher.method,
        'pixel_match': stitcher.pixel_match,
        'lab_match': stitcher.lab_match,
        'multi_band': stitcher.multi_band,
        'mbb_num_bands': getattr(stitcher, 'mbb_num_bands', 5),
        '_mbb_num_bands': getattr(stitcher, '_mbb_num_bands', 0),
        '_mbb_alpha_pyr': getattr(stitcher, '_mbb_alpha_pyr', None),
        '_mbb_valid_mask_3c': getattr(stitcher, '_mbb_valid_mask_3c', None),
        'image_size': stitcher.image_size,
        'w': stitcher.w, 'h': stitcher.h,
        'focal_length': stitcher.focal_length,
        'H': stitcher.H,
        'canvas_w': stitcher.canvas_w, 'canvas_h': stitcher.canvas_h,
        'offset_x': stitcher.offset_x, 'offset_y': stitcher.offset_y,
        'T': stitcher.T, 'H_translated': stitcher.H_translated,
        '_cyl_maps': stitcher._cyl_maps,
        '_left_combined_maps': stitcher._left_combined_maps,
        '_right_combined_maps': stitcher._right_combined_maps,
        '_mask_left_only_3c': stitcher._mask_left_only_3c,
        '_mask_right_only_3c': stitcher._mask_right_only_3c,
        '_overlap_mask_3c': stitcher._overlap_mask_3c,
        '_blend_alpha_3c': stitcher._blend_alpha_3c,
        '_has_overlap': stitcher._has_overlap,
        '_crop_rect': stitcher._crop_rect,
        '_exposure_luts': stitcher._exposure_luts,
        '_ab_shifts': stitcher._ab_shifts,
        '_gpu_maps': None,  # GPU는 프로세스 간 공유 불가
    }

    process_args = [
        (i, left, right, stitcher_params, output)
        for i, (left, right, output) in enumerate(frame_pairs)
    ]

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"\nProcessing with {num_workers} workers...")

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(frame_pairs), desc="Stitching")
    except ImportError:
        progress = None

    success_count = 0
    error_count = 0

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_frame, arg): arg[0]
                       for arg in process_args}
            for future in as_completed(futures):
                idx, success, error = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\nError on frame {idx}: {error}")
                if progress:
                    progress.update(1)
    else:
        for arg in process_args:
            idx, success, error = process_single_frame(arg)
            if success:
                success_count += 1
            else:
                error_count += 1
                print(f"\nError on frame {idx}: {error}")
            if progress:
                progress.update(1)
            else:
                print(f"\rProcessed {idx + 1}/{len(frame_pairs)}", end='')

    if progress:
        progress.close()

    print(f"\n\nCompleted: {success_count} success, {error_count} errors")
    print(f"Output directory: {output_dir}")
    return output_dir


# ============================================================================
# STREAMING STITCH (IN-MEMORY PIPELINE)
# ============================================================================

def stitch_from_iterator(frame_iterator, calibration_path, output_dir,
                         method='cylindrical', focal_weight='auto', fps=30,
                         no_video=False, total_frames=None, audio_source=None,
                         lab_match=True, pixel_match=True, multi_band=True):
    """메모리 이터레이터에서 프레임을 받아 스티칭한다 (디스크 I/O 최소화).

    Args:
        frame_iterator: (left_frame, right_frame) BGR numpy 쌍을 yield
        calibration_path: calibration JSON 경로
        output_dir: 출력 디렉토리
        method: 'cylindrical' 또는 'planar'
        focal_weight: focal_length 가중치
        fps: 출력 비디오 FPS
        no_video: True면 비디오 생성 스킵
        total_frames: 총 프레임 수 (tqdm 진행률 표시용, None이면 미표시)
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    stitcher = PanoramaStitcher(calibration_path, method=method,
                                focal_weight=focal_weight,
                                lab_match=lab_match, pixel_match=pixel_match,
                                multi_band=multi_band)

    try:
        from tqdm import tqdm
        progress = tqdm(total=total_frames, desc="Stitching")
    except ImportError:
        progress = None

    count = 0
    for left_img, right_img in frame_iterator:
        result = stitcher.stitch(left_img, right_img, crop=True)
        out_path = frames_dir / f"stitched_frame_{count + 1:06d}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
        if progress:
            progress.update(1)
        elif count % 100 == 0:
            print(f"\r  Stitched {count} frames...", end='')

    if progress:
        progress.close()
    print(f"\n  Stitched {count} frames total")

    if not no_video and count > 0:
        create_video(frames_dir, output_dir / 'panorama.mp4', fps=fps,
                     audio_source=audio_source)

    return output_dir


# ============================================================================
# VIDEO
# ============================================================================

def create_video(frames_dir, output_video, fps=30, codec='mp4v', audio_source=None):
    """스티칭된 프레임들로 영상을 생성한다.

    Args:
        audio_source: 오디오를 가져올 원본 영상 경로 (예: left.MOV). None이면 무음.
    """
    frames_dir = Path(frames_dir)
    output_video = Path(output_video)

    frames = sorted(frames_dir.glob('stitched_*.jpg'))
    if not frames:
        frames = sorted(frames_dir.glob('*.jpg'))
    if not frames:
        print("No frames found!")
        return

    print(f"Creating video from {len(frames)} frames...")

    first_frame = cv2.imread(str(frames[0]))
    h, w = first_frame.shape[:2]
    print(f"Frame size: {w}x{h}, FPS: {fps}, Duration: {len(frames)/fps:.1f}s")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    try:
        from tqdm import tqdm
        frames_iter = tqdm(frames, desc="Writing video")
    except ImportError:
        frames_iter = frames

    for frame_path in frames_iter:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out.write(frame)

    out.release()
    print(f"Video saved: {output_video}")

    try:
        import subprocess
        web_output = output_video.with_suffix('.web.mp4')

        if audio_source and Path(audio_source).exists():
            # 오디오 포함: 원본 영상에서 오디오 추출하여 합치기
            cmd = ['ffmpeg', '-y',
                   '-i', str(output_video),
                   '-i', str(audio_source),
                   '-c:v', 'libx264', '-preset', 'medium',
                   '-crf', '23', '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac', '-b:a', '192k',
                   '-map', '0:v:0', '-map', '1:a:0',
                   '-shortest',
                   '-movflags', '+faststart', str(web_output)]
            print(f"Muxing audio from: {audio_source}")
        else:
            # 무음 비디오
            cmd = ['ffmpeg', '-y', '-i', str(output_video),
                   '-c:v', 'libx264', '-preset', 'medium',
                   '-crf', '23', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(web_output)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            has_audio = audio_source and Path(audio_source).exists()
            print(f"Web-compatible video saved: {web_output}" +
                  (" (with audio)" if has_audio else " (no audio)"))
        else:
            print(f"FFmpeg error: {result.stderr[:200]}")
    except Exception as e:
        print(f"FFmpeg failed: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Panorama Video Stitching Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stitch_video.py --left ./left --right ./right --calib calib.json
  python stitch_video.py --left ./left --right ./right --calib calib.json --focal-weight 0.8
  python stitch_video.py --video-only --frames ./output/frames --fps 30
        """)

    parser.add_argument('--left', type=str, help='Left camera frames directory')
    parser.add_argument('--right', type=str, help='Right camera frames directory')
    parser.add_argument('--calib', type=str, help='Calibration JSON file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--method', type=str, default='cylindrical',
                        choices=['cylindrical', 'planar'], help='Stitching method')
    parser.add_argument('--focal-weight', default='auto',
                        help='focal_length = width * weight (default: auto)')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS')
    parser.add_argument('--no-video', action='store_true', help='Skip video creation')
    parser.add_argument('--video-only', action='store_true',
                        help='Only create video from existing frames')
    parser.add_argument('--frames', type=str, help='Frames directory (--video-only)')
    parser.add_argument('--audio', type=str, default=None,
                        help='Audio source video file (e.g. left.MOV) to mux into output')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / 'frames'

    if args.video_only:
        if args.frames:
            frames_dir = Path(args.frames)
        create_video(frames_dir, output_dir / 'panorama.mp4', fps=args.fps,
                     audio_source=args.audio)
    else:
        if not all([args.left, args.right, args.calib]):
            parser.error("--left, --right, --calib are required for stitching")

        fw = args.focal_weight
        if fw != 'auto':
            fw = float(fw)

        batch_stitch(
            args.left, args.right, args.calib, frames_dir,
            method=args.method, num_workers=args.workers, focal_weight=fw)

        if not args.no_video:
            audio_src = args.audio or args.left
            create_video(frames_dir, output_dir / 'panorama.mp4', fps=args.fps,
                         audio_source=audio_src)
            print(f"\nPipeline complete!")
            print(f"  Frames: {frames_dir}")
            print(f"  Video: {output_dir / 'panorama.mp4'}")


if __name__ == '__main__':
    main()
