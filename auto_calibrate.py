"""
Auto Calibration -DISK + LightGlue 자동 대응점 매칭
카메라 위치가 바뀔 때마다 수동 없이 calibration.json 생성
"""

import cv2
import numpy as np
import json
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SuperPoint + LightGlue 매칭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_and_match_lightglue(left, right):
    """DISK + LightGlue 매칭"""
    import torch
    from kornia.feature import DISK
    from kornia.feature.lightglue import LightGlue

    device = torch.device("cpu")

    # BGR → RGB float tensor (N, 3, H, W)
    def to_tensor(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0

    t_left = to_tensor(left).to(device)
    t_right = to_tensor(right).to(device)
    h, w = left.shape[:2]

    # DISK 특징점 검출 (더 많이 검출)
    disk = DISK.from_pretrained("depth").to(device).eval()
    print("  DISK 특징점 검출 중...")
    with torch.no_grad():
        feats0 = disk(t_left, n=8192, pad_if_not_divisible=True)[0]
        feats1 = disk(t_right, n=8192, pad_if_not_divisible=True)[0]

    print(f"  좌측: {feats0.keypoints.shape[0]}, 우측: {feats1.keypoints.shape[0]}")

    # 원본 이미지 범위 밖 키포인트 필터 + 경계에서 약간 안쪽으로 클램프
    # (패딩으로 인해 범위 넘는 키포인트 방지)
    def clamp_feats(feats, h, w):
        kp = feats.keypoints
        valid = (kp[:, 0] >= 0) & (kp[:, 0] < w) & (kp[:, 1] >= 0) & (kp[:, 1] < h)
        from kornia.feature.disk.structs import DISKFeatures
        return DISKFeatures(
            keypoints=kp[valid],
            descriptors=feats.descriptors[valid],
            detection_scores=feats.detection_scores[valid],
        )

    feats0 = clamp_feats(feats0, h, w)
    feats1 = clamp_feats(feats1, h, w)
    print(f"  범위 필터 후 -좌측: {feats0.keypoints.shape[0]}, 우측: {feats1.keypoints.shape[0]}")

    # LightGlue 매칭
    lg = LightGlue("disk").to(device).eval()
    print("  LightGlue 매칭 중...")

    # LightGlue가 내부에서 normalize_keypoints를 호출하므로 원본 픽셀 좌표 그대로 전달
    input_dict = {
        "image0": {
            "keypoints": feats0.keypoints[None],       # (1, N, 2) 픽셀 좌표
            "descriptors": feats0.descriptors[None],   # (1, N, 128)
            "image_size": torch.tensor([[w, h]], device=device),
        },
        "image1": {
            "keypoints": feats1.keypoints[None],
            "descriptors": feats1.descriptors[None],
            "image_size": torch.tensor([[w, h]], device=device),
        },
    }

    with torch.no_grad():
        result = lg(input_dict)

    print(f"  LightGlue output keys: {list(result.keys())}")

    # 버전에 따라 key 이름이 다름
    matches_key = "matches" if "matches" in result else "matches0"
    scores_key = next((k for k in ("matching_scores", "scores", "match_scores") if k in result), None)

    matches = result[matches_key][0].cpu().numpy()

    if scores_key:
        scores = result[scores_key][0].cpu().numpy()
    else:
        scores = np.ones(len(matches))

    # matches가 (N0,) 형태: matches[i] = kp1 인덱스 (or -1)
    if matches.ndim == 1:
        valid = matches >= 0
        idx0 = np.where(valid)[0]
        idx1 = matches[valid]
        conf = scores[valid] if scores_key else np.ones(idx0.shape[0])
    else:
        # (M, 2) 형태
        valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
        idx0 = matches[valid, 0]
        idx1 = matches[valid, 1]
        conf = scores[valid] if scores_key else np.ones(idx0.shape[0])

    kp0 = feats0.keypoints.cpu().numpy()[idx0]
    kp1 = feats1.keypoints.cpu().numpy()[idx1]

    print(f"  매칭 수: {len(kp0)}")

    if len(kp0) < 4:
        print("ERROR: 매칭이 부족합니다.")
        return [], None

    # 고신뢰도 순 정렬
    sorted_idx = np.argsort(-conf)
    kp0 = kp0[sorted_idx]
    kp1 = kp1[sorted_idx]
    conf = conf[sorted_idx]

    # RANSAC
    H, mask = cv2.findHomography(kp1, kp0, cv2.RANSAC, 5.0)
    if H is None:
        print("ERROR: 호모그래피 계산 실패")
        return [], None

    inliers = mask.ravel().astype(bool)
    kp0 = kp0[inliers]
    kp1 = kp1[inliers]
    conf = conf[inliers]

    print(f"  RANSAC 인라이어: {inliers.sum()}/{len(inliers)}")

    # 재투영 오차
    kp1_h = np.hstack([kp1, np.ones((len(kp1), 1))])
    projected = (H @ kp1_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - kp0, axis=1)

    # 오차 작은 순 정렬, 상위 유지
    sorted_idx = np.argsort(errors)
    max_keep = min(80, len(sorted_idx))
    kp0 = kp0[sorted_idx[:max_keep]]
    kp1 = kp1[sorted_idx[:max_keep]]
    errors = errors[sorted_idx[:max_keep]]

    print(f"  재투영 오차 -평균: {errors.mean():.2f}px, 최대: {errors.max():.2f}px")

    # 공간 분포 최적화 (16x16 그리드 → 더 많은 셀에서 포인트 유지)
    kp0, kp1 = spatial_subsample(kp0, kp1, left.shape, grid_size=16)
    print(f"  공간 분포 최적화 후: {len(kp0)}쌍")

    # 최종 호모그래피 재계산
    if len(kp0) >= 4:
        H, _ = cv2.findHomography(kp1, kp0, cv2.RANSAC, 5.0)

    point_pairs = []
    for i in range(len(kp0)):
        point_pairs.append({
            "left": [round(float(kp0[i][0]), 1), round(float(kp0[i][1]), 1)],
            "right": [round(float(kp1[i][0]), 1), round(float(kp1[i][1]), 1)],
            "source": "disk_lightglue_auto"
        })

    return point_pairs, H


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def spatial_subsample(pts_left, pts_right, img_shape, grid_size=8):
    """겹침 영역을 그리드로 나눠 각 셀에서 가장 좋은 1개씩 선택"""
    h, w = img_shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    grid = {}
    for i in range(len(pts_left)):
        gx = int(pts_left[i][0] / cell_w)
        gy = int(pts_left[i][1] / cell_h)
        key = (gx, gy)
        if key not in grid:
            grid[key] = i

    indices = list(grid.values())
    return pts_left[indices], pts_right[indices]


def visualize(left, right, point_pairs, output_path):
    """매칭 결과 시각화"""
    h, w = left.shape[:2]
    combined = np.hstack([left, right])

    colors = [
        (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (0, 165, 255), (255, 0, 0), (0, 0, 255), (128, 255, 0),
        (255, 128, 0), (0, 255, 128), (255, 0, 128), (128, 0, 255),
    ]

    for i, pair in enumerate(point_pairs):
        lx, ly = int(pair["left"][0]), int(pair["left"][1])
        rx, ry = int(pair["right"][0]) + w, int(pair["right"][1])
        color = colors[i % len(colors)]

        cv2.line(combined, (lx, ly), (rx, ry), color, 1, cv2.LINE_AA)
        cv2.circle(combined, (lx, ly), 6, color, 2, cv2.LINE_AA)
        cv2.circle(combined, (rx, ry), 6, color, 2, cv2.LINE_AA)
        cv2.putText(combined, str(i + 1), (lx + 8, ly - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"시각화 저장: {output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="자동 캘리브레이션 (DISK + LightGlue)")
    parser.add_argument("--left", default=os.path.join(BASE_DIR, "left_sync/frame_00001.jpg"))
    parser.add_argument("--right", default=os.path.join(BASE_DIR, "right_sync/frame_00001.jpg"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "calibration_auto.json"))
    parser.add_argument("--left-focal", type=float, default=None,
                        help="좌측 카메라 35mm 환산 focal length (mm)")
    parser.add_argument("--right-focal", type=float, default=None,
                        help="우측 카메라 35mm 환산 focal length (mm)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Auto Calibration (DISK + LightGlue)")
    print("=" * 60)

    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    if left is None or right is None:
        print(f"ERROR: 이미지를 불러올 수 없습니다.")
        print(f"  left:  {args.left}")
        print(f"  right: {args.right}")
        sys.exit(1)

    print(f"\n이미지: {left.shape[1]}x{left.shape[0]}")
    print(f"\n매칭 중 (DISK + LightGlue)...")

    point_pairs, H = detect_and_match_lightglue(left, right)

    if not point_pairs:
        print("\nERROR: 매칭 실패")
        sys.exit(1)

    calib = {
        "point_pairs": point_pairs,
        "image_size": [left.shape[1], left.shape[0]],
        "left_image": os.path.relpath(args.left, BASE_DIR),
        "right_image": os.path.relpath(args.right, BASE_DIR),
        "method": "lightglue",
    }

    if args.left_focal is not None:
        calib["left_focal_mm"] = args.left_focal
    if args.right_focal is not None:
        calib["right_focal_mm"] = args.right_focal

    if H is not None:
        calib["homography"] = H.tolist()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {args.output} ({len(point_pairs)}쌍)")

    # 매칭된 특징점 시각화 저장
    vis_path = args.output.replace(".json", "_matches_vis.jpg")
    visualize(left, right, point_pairs, vis_path)

    print("\n" + "=" * 60)
    print(f"  완료! {len(point_pairs)}쌍 자동 매칭 (DISK + LightGlue)")
    print("=" * 60)


if __name__ == "__main__":
    main()
