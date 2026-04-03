#!/usr/bin/env python3
"""
Panorama Video Stitching Pipeline
==================================
Stitches synchronized left/right camera frames into panoramic video.

Usage:
    python stitch_video.py --left ./left_sync --right ./right_sync --calib calibration.json --output ./output

Requirements:
    pip install opencv-python numpy tqdm

Author: Claude
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ============================================================================
# CYLINDRICAL PROJECTION FUNCTIONS
# ============================================================================

def build_cylindrical_maps(h, w, focal_length):
    """Precompute cylindrical remap maps (call once, reuse for every frame)."""
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
    """Apply cylindrical projection to an image."""
    if maps is None:
        h, w = img.shape[:2]
        maps = build_cylindrical_maps(h, w, focal_length)
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def planar_to_cylindrical(x, y, focal_length, cx, cy):
    """Convert planar coordinates to cylindrical coordinates."""
    theta = np.arctan((x - cx) / focal_length)
    x_cyl = focal_length * theta + cx
    y_cyl = (y - cy) * np.cos(theta) + cy
    return x_cyl, y_cyl


def get_valid_bounds(img):
    """Get bounding box of non-black pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.where(gray > 0)
    if len(coords[0]) == 0:
        return 0, img.shape[0]-1, 0, img.shape[1]-1
    return coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()


# ============================================================================
# STITCHER CLASS
# ============================================================================

class PanoramaStitcher:
    """Handles stitching of left/right image pairs."""
    
    def __init__(self, calibration_path, method='cylindrical'):
        """
        Initialize stitcher with calibration data.
        
        Args:
            calibration_path: Path to calibration.json
            method: 'cylindrical' or 'planar'
        """
        with open(calibration_path, 'r') as f:
            self.calib = json.load(f)
        
        self.method = method
        self.image_size = tuple(self.calib['image_size'])
        self.w, self.h = self.image_size
        
        # Focal length estimation (can be adjusted)
        self.focal_length = self.w * 1.0

        # Precompute cylindrical remap maps (reused every frame)
        self._cyl_maps = None
        if method == 'cylindrical':
            self._cyl_maps = build_cylindrical_maps(self.h, self.w, self.focal_length)

        # Precompute homography
        self._compute_homography()

        # Precompute canvas parameters
        self._compute_canvas_params()

        # Precompute blend weight (same for every frame)
        self._precompute_blend_weight()
        
        print(f"Stitcher initialized: {method} method")
        print(f"  Image size: {self.w}x{self.h}")
        print(f"  Focal length: {self.focal_length:.0f}")
        print(f"  Canvas size: {self.canvas_w}x{self.canvas_h}")
    
    def _compute_homography(self):
        """Compute homography matrix for stitching."""
        cx, cy = self.w / 2, self.h / 2
        
        if self.method == 'cylindrical':
            # Transform point pairs to cylindrical coordinates
            cyl_points_left = []
            cyl_points_right = []
            
            for pp in self.calib['point_pairs']:
                lx, ly = pp['left']
                rx, ry = pp['right']
                
                lx_c, ly_c = planar_to_cylindrical(lx, ly, self.focal_length, cx, cy)
                rx_c, ry_c = planar_to_cylindrical(rx, ry, self.focal_length, cx, cy)
                
                cyl_points_left.append([lx_c, ly_c])
                cyl_points_right.append([rx_c, ry_c])
            
            cyl_points_left = np.array(cyl_points_left, dtype=np.float32)
            cyl_points_right = np.array(cyl_points_right, dtype=np.float32)
            
            self.H, mask = cv2.findHomography(cyl_points_right, cyl_points_left, cv2.RANSAC, 5.0)
        else:
            # Use planar homography from calibration
            self.H = np.array(self.calib['homography'], dtype=np.float64)
    
    def _compute_canvas_params(self):
        """Precompute canvas size and offset for consistent output."""
        # Create dummy images to compute bounds
        dummy = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        
        if self.method == 'cylindrical':
            cyl_img = cylindrical_warp(dummy, self.focal_length)
            y1, y2, x1, x2 = get_valid_bounds(cyl_img)
        else:
            y1, y2, x1, x2 = 0, self.h-1, 0, self.w-1
        
        # Corners for right image
        corners_right = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        warped_corners = cv2.perspectiveTransform(corners_right, self.H).reshape(-1, 2)
        
        # Corners for left image
        corners_left = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)
        
        all_corners = np.vstack([corners_left, warped_corners])
        
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
        
        self.canvas_w = x_max - x_min
        self.canvas_h = y_max - y_min
        self.offset_x = -x_min
        self.offset_y = -y_min
        
        # Translation matrix
        self.T = np.array([
            [1, 0, self.offset_x],
            [0, 1, self.offset_y],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.H_translated = self.T @ self.H

    def _precompute_blend_weight(self):
        """Precompute blend weight map and masks using dummy white images.

        The overlap region is identical for every frame (same camera setup),
        so we compute it once and reuse.
        """
        dummy = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255

        if self._cyl_maps is not None:
            dummy = cylindrical_warp(dummy, self.focal_length, maps=self._cyl_maps)

        warped_right = cv2.warpPerspective(dummy, self.H_translated,
                                           (self.canvas_w, self.canvas_h))
        canvas_left = cv2.warpPerspective(dummy, self.T,
                                          (self.canvas_w, self.canvas_h))

        mask_left = (canvas_left > 0).any(axis=2).astype(np.float32)
        mask_right = (warped_right > 0).any(axis=2).astype(np.float32)
        overlap = mask_left * mask_right

        # Vectorized per-row linear blend weight
        blend_weight = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
        for y in range(self.canvas_h):
            cols = np.where(overlap[y, :] > 0)[0]
            if len(cols) > 1:
                x_start, x_end = cols[0], cols[-1]
                width = x_end - x_start + 1
                blend_weight[y, x_start:x_end + 1] = np.linspace(0, 1, width)

        # Cache as 3D arrays for vectorized compositing
        self._mask_left_only = (mask_left * (1 - mask_right))[:, :, np.newaxis]
        self._mask_right_only = (mask_right * (1 - mask_left))[:, :, np.newaxis]
        self._blend_left = ((1 - blend_weight) * overlap)[:, :, np.newaxis]
        self._blend_right = (blend_weight * overlap)[:, :, np.newaxis]

        # Crop bounds from the dummy composite
        composite = np.clip(
            canvas_left * self._mask_left_only
            + warped_right * self._mask_right_only
            + canvas_left * self._blend_left
            + warped_right * self._blend_right,
            0, 255,
        ).astype(np.uint8)
        gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is not None:
            self._crop_rect = cv2.boundingRect(coords)  # x, y, w, h
        else:
            self._crop_rect = None

    def stitch(self, left_img, right_img, crop=True):
        """
        Stitch left and right images.

        Args:
            left_img: Left camera image (BGR)
            right_img: Right camera image (BGR)
            crop: Whether to crop black borders

        Returns:
            Stitched panorama image
        """
        if self._cyl_maps is not None:
            left_proc = cylindrical_warp(left_img, self.focal_length, maps=self._cyl_maps)
            right_proc = cylindrical_warp(right_img, self.focal_length, maps=self._cyl_maps)
        else:
            left_proc = left_img
            right_proc = right_img

        # Warp images onto canvas
        warped_right = cv2.warpPerspective(
            right_proc, self.H_translated, (self.canvas_w, self.canvas_h)
        )
        canvas_left = cv2.warpPerspective(
            left_proc, self.T, (self.canvas_w, self.canvas_h)
        )

        # Vectorized composite using precomputed masks (no per-frame loop)
        canvas_left_f = canvas_left.astype(np.float32)
        warped_right_f = warped_right.astype(np.float32)

        result = (canvas_left_f * self._mask_left_only
                  + warped_right_f * self._mask_right_only
                  + canvas_left_f * self._blend_left
                  + warped_right_f * self._blend_right)

        result = np.clip(result, 0, 255).astype(np.uint8)

        if crop and self._crop_rect is not None:
            x, y, rw, rh = self._crop_rect
            result = result[y:y+rh, x:x+rw]

        return result


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_single_frame(args):
    """Process a single frame pair (for multiprocessing)."""
    idx, left_path, right_path, stitcher_params, output_path = args
    
    # Recreate stitcher in subprocess
    stitcher = PanoramaStitcher.__new__(PanoramaStitcher)
    stitcher.__dict__.update(stitcher_params)
    
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    
    if left_img is None or right_img is None:
        return idx, False, f"Failed to load images"
    
    try:
        result = stitcher.stitch(left_img, right_img, crop=True)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


def batch_stitch(left_dir, right_dir, calibration_path, output_dir, 
                 method='cylindrical', num_workers=None, frame_pattern='frame_*.jpg'):
    """
    Batch stitch all frame pairs.
    
    Args:
        left_dir: Directory containing left camera frames
        right_dir: Directory containing right camera frames
        calibration_path: Path to calibration.json
        output_dir: Directory for output stitched frames
        method: 'cylindrical' or 'planar'
        num_workers: Number of parallel workers (default: CPU count)
        frame_pattern: Glob pattern for frame files
    """
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frame pairs
    left_frames = sorted(left_dir.glob(frame_pattern))
    
    if not left_frames:
        # Try alternative patterns
        for pattern in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
            left_frames = sorted(left_dir.glob(pattern))
            if left_frames:
                break
    
    print(f"Found {len(left_frames)} frames in left directory")
    
    # Match with right frames
    frame_pairs = []
    for left_path in left_frames:
        # Try to find matching right frame
        right_path = right_dir / left_path.name
        if not right_path.exists():
            # Try replacing 'left' with 'right' in filename
            alt_name = left_path.name.replace('left', 'right').replace('LEFT', 'RIGHT')
            right_path = right_dir / alt_name
        
        if right_path.exists():
            output_path = output_dir / f"stitched_{left_path.stem}.jpg"
            frame_pairs.append((left_path, right_path, output_path))
        else:
            print(f"Warning: No matching right frame for {left_path.name}")
    
    print(f"Matched {len(frame_pairs)} frame pairs")
    
    if not frame_pairs:
        print("No frame pairs found!")
        return
    
    # Initialize stitcher to get parameters
    stitcher = PanoramaStitcher(calibration_path, method=method)
    
    # Serialize stitcher parameters for multiprocessing
    stitcher_params = {
        'method': stitcher.method,
        'image_size': stitcher.image_size,
        'w': stitcher.w,
        'h': stitcher.h,
        'focal_length': stitcher.focal_length,
        'H': stitcher.H,
        'canvas_w': stitcher.canvas_w,
        'canvas_h': stitcher.canvas_h,
        'offset_x': stitcher.offset_x,
        'offset_y': stitcher.offset_y,
        'T': stitcher.T,
        'H_translated': stitcher.H_translated,
        '_cyl_maps': stitcher._cyl_maps,
        '_mask_left_only': stitcher._mask_left_only,
        '_mask_right_only': stitcher._mask_right_only,
        '_blend_left': stitcher._blend_left,
        '_blend_right': stitcher._blend_right,
        '_crop_rect': stitcher._crop_rect,
    }
    
    # Prepare arguments for parallel processing
    process_args = [
        (i, left, right, stitcher_params, output)
        for i, (left, right, output) in enumerate(frame_pairs)
    ]
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"\nProcessing with {num_workers} workers...")
    
    # Process frames
    try:
        from tqdm import tqdm
        progress = tqdm(total=len(frame_pairs), desc="Stitching")
    except ImportError:
        progress = None
        print("(Install tqdm for progress bar: pip install tqdm)")
    
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
        # Single-threaded processing
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
                print(f"\rProcessed {idx+1}/{len(frame_pairs)}", end='')
    
    if progress:
        progress.close()
    
    print(f"\n\nCompleted: {success_count} success, {error_count} errors")
    print(f"Output directory: {output_dir}")
    
    return output_dir


def create_video(frames_dir, output_video, fps=30, codec='mp4v'):
    """
    Create video from stitched frames.
    
    Args:
        frames_dir: Directory containing stitched frames
        output_video: Output video path
        fps: Frames per second
        codec: Video codec (mp4v, XVID, H264, etc.)
    """
    frames_dir = Path(frames_dir)
    output_video = Path(output_video)
    
    # Find all frames
    frames = sorted(frames_dir.glob('stitched_*.jpg'))
    if not frames:
        frames = sorted(frames_dir.glob('*.jpg'))
    
    if not frames:
        print("No frames found!")
        return
    
    print(f"Creating video from {len(frames)} frames...")
    
    # Get frame dimensions from first frame
    first_frame = cv2.imread(str(frames[0]))
    h, w = first_frame.shape[:2]
    
    print(f"Frame size: {w}x{h}")
    print(f"FPS: {fps}")
    print(f"Duration: {len(frames)/fps:.1f} seconds")
    
    # Create video writer
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
            # Resize if needed (some frames might have slightly different sizes)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out.write(frame)
    
    out.release()
    print(f"Video saved: {output_video}")
    
    # Try to create web-compatible version with ffmpeg
    try:
        import subprocess
        web_output = output_video.with_suffix('.web.mp4')
        cmd = [
            'ffmpeg', '-y', '-i', str(output_video),
            '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(web_output)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Web-compatible video saved: {web_output}")
    except Exception as e:
        print(f"(ffmpeg not available for web video conversion: {e})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Panorama Video Stitching Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python stitch_video.py --left ./left_sync --right ./right_sync --calib calibration.json

  # With custom output and planar method
  python stitch_video.py --left ./left --right ./right --calib calib.json --output ./panoramas --method planar

  # Create video only (from existing stitched frames)
  python stitch_video.py --video-only --frames ./output/frames --fps 30
        """
    )
    
    parser.add_argument('--left', type=str, help='Left camera frames directory')
    parser.add_argument('--right', type=str, help='Right camera frames directory')
    parser.add_argument('--calib', type=str, help='Calibration JSON file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--method', type=str, default='cylindrical', 
                       choices=['cylindrical', 'planar'], help='Stitching method')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS')
    parser.add_argument('--no-video', action='store_true', help='Skip video creation')
    parser.add_argument('--video-only', action='store_true', help='Only create video from existing frames')
    parser.add_argument('--frames', type=str, help='Frames directory (for --video-only)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / 'frames'
    
    if args.video_only:
        # Only create video
        if args.frames:
            frames_dir = Path(args.frames)
        create_video(frames_dir, output_dir / 'panorama.mp4', fps=args.fps)
    else:
        # Full pipeline
        if not all([args.left, args.right, args.calib]):
            parser.error("--left, --right, and --calib are required for stitching")
        
        # Batch stitch
        batch_stitch(
            args.left, args.right, args.calib,
            frames_dir,
            method=args.method,
            num_workers=args.workers
        )
        
        # Create video
        if not args.no_video:
            create_video(frames_dir, output_dir / 'panorama.mp4', fps=args.fps)
            print(f"\nPipeline complete!")
            print(f"  Frames: {frames_dir}")
            print(f"  Video: {output_dir / 'panorama.mp4'}")


if __name__ == '__main__':
    main()
