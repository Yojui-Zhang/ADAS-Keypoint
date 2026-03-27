import os
import glob
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass
class DetectedFrame:
    source_path: str
    frame_index: int
    image_size: Tuple[int, int]          # (w, h)
    corners: np.ndarray                  # (N,1,2) float32
    sharpness: float
    coverage: float                      # 0~1 approximate coverage ratio
    saved_path: Optional[str] = None


def is_image_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


def is_video_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in VID_EXTS


def list_media_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]

    files = []
    for root, _, names in os.walk(input_path):
        for n in names:
            p = os.path.join(root, n)
            if is_image_file(p) or is_video_file(p):
                files.append(p)
    files.sort()
    return files


def compute_sharpness(gray: np.ndarray) -> float:
    # Variance of Laplacian: higher generally means sharper
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_coverage(corners: np.ndarray, w: int, h: int) -> float:
    # approximate coverage by bounding box area ratio
    xy = corners.reshape(-1, 2)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    area = max(0.0, (xmax - xmin)) * max(0.0, (ymax - ymin))
    return float(area / (w * h + 1e-9))


def find_chessboard_corners(
    img_bgr: np.ndarray,
    board_cols: int,
    board_rows: int,
    subpix: bool = True
) -> Optional[np.ndarray]:
    """
    board_cols/board_rows: number of inner corners per chessboard row/col
    returns corners: (N,1,2) float32 in pixel coords
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    pattern_size = (board_cols, board_rows)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)

    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        # Try SB detector (often better for difficult images, OpenCV >= 4.5+)
        try:
            ok2, corners2 = cv2.findChessboardCornersSB(gray, pattern_size)
            if ok2:
                corners = corners2
                ok = True
        except Exception:
            pass

    if not ok:
        return None

    corners = corners.astype(np.float32)

    if subpix:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-6)
        cv2.cornerSubPix(
            gray, corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )
    return corners


def extract_from_image(
    path: str,
    board_cols: int,
    board_rows: int,
    save_used_dir: Optional[str]
) -> Optional[DetectedFrame]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    corners = find_chessboard_corners(img, board_cols, board_rows, subpix=True)
    if corners is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = compute_sharpness(gray)
    cov = compute_coverage(corners, w, h)

    saved = None
    if save_used_dir:
        os.makedirs(save_used_dir, exist_ok=True)
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (board_cols, board_rows), corners, True)
        base = os.path.splitext(os.path.basename(path))[0]
        saved = os.path.join(save_used_dir, f"{base}_used.png")
        cv2.imwrite(saved, vis)

    return DetectedFrame(
        source_path=path,
        frame_index=-1,
        image_size=(w, h),
        corners=corners,
        sharpness=sharp,
        coverage=cov,
        saved_path=saved
    )


def extract_from_video(
    path: str,
    board_cols: int,
    board_rows: int,
    sample_every_n_frames: int,
    max_frames_to_process: int,
    save_used_dir: Optional[str]
) -> List[DetectedFrame]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    used = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_every_n_frames != 0:
            idx += 1
            continue

        processed += 1
        if processed > max_frames_to_process:
            break

        h, w = frame.shape[:2]
        corners = find_chessboard_corners(frame, board_cols, board_rows, subpix=True)
        if corners is None:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = compute_sharpness(gray)
        cov = compute_coverage(corners, w, h)

        saved = None
        if save_used_dir:
            os.makedirs(save_used_dir, exist_ok=True)
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, (board_cols, board_rows), corners, True)
            base = os.path.splitext(os.path.basename(path))[0]
            saved = os.path.join(save_used_dir, f"{base}_frame{idx:06d}_used.png")
            cv2.imwrite(saved, vis)

        used.append(DetectedFrame(
            source_path=path,
            frame_index=idx,
            image_size=(w, h),
            corners=corners,
            sharpness=sharp,
            coverage=cov,
            saved_path=saved
        ))

        idx += 1

    cap.release()
    return used


def build_object_points(board_cols: int, board_rows: int, square_size: float) -> np.ndarray:
    """
    Generate Nx3 object points for chessboard inner corners on Z=0 plane.
    square_size unit can be meters or any consistent unit.
    """
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def reprojection_errors(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    D: np.ndarray
) -> Tuple[List[float], float]:
    per_view = []
    total_sq = 0.0
    total_n = 0

    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        obs = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(proj - obs, axis=1)
        rmse = float(np.sqrt(np.mean(err**2)))
        per_view.append(rmse)

        total_sq += float(np.sum(err**2))
        total_n += err.shape[0]

    global_rmse = float(np.sqrt(total_sq / max(1, total_n)))
    return per_view, global_rmse


def calibrate_intrinsics(
    detections: List[DetectedFrame],
    board_cols: int,
    board_rows: int,
    square_size: float,
    use_rational_model: bool,
    do_outlier_rejection: bool,
    outlier_sigma: float
):
    if len(detections) < 8:
        raise RuntimeError(f"Not enough valid frames. Need at least ~8, got {len(detections)}")

    # Ensure consistent image size
    w0, h0 = detections[0].image_size
    for d in detections:
        if d.image_size != (w0, h0):
            raise RuntimeError("Image sizes are not consistent. Please calibrate with a single resolution.")

    objp = build_object_points(board_cols, board_rows, square_size)

    objpoints = [objp.copy() for _ in detections]
    imgpoints = [d.corners.copy() for d in detections]

    flags = 0
    # Most common: keep tangential distortion enabled (p1,p2)
    # Optionally add rational model (k4,k5,k6) which can improve for some lenses
    if use_rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)

    # First pass
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w0, h0),
        None, None,
        flags=flags,
        criteria=criteria
    )

    per_view, global_rmse = reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, D)

    kept_indices = list(range(len(detections)))

    # Outlier rejection + second pass
    if do_outlier_rejection and len(detections) >= 12:
        mu = float(np.mean(per_view))
        sd = float(np.std(per_view) + 1e-9)
        thr = mu + outlier_sigma * sd

        new_kept = [i for i, e in enumerate(per_view) if e <= thr]

        # Ensure we keep enough frames
        if len(new_kept) >= 8 and len(new_kept) < len(detections):
            kept_indices = new_kept
            objpoints2 = [objpoints[i] for i in kept_indices]
            imgpoints2 = [imgpoints[i] for i in kept_indices]

            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints2, imgpoints2, (w0, h0),
                None, None,
                flags=flags,
                criteria=criteria
            )
            per_view2, global_rmse2 = reprojection_errors(objpoints2, imgpoints2, rvecs, tvecs, K, D)

            return {
                "K": K, "D": D,
                "rms": float(rms),
                "global_rmse": float(global_rmse2),
                "per_view_rmse": per_view2,
                "kept_indices": kept_indices,
                "image_size": (w0, h0),
                "flags": flags
            }

    return {
        "K": K, "D": D,
        "rms": float(rms),
        "global_rmse": float(global_rmse),
        "per_view_rmse": per_view,
        "kept_indices": kept_indices,
        "image_size": (w0, h0),
        "flags": flags
    }


def write_opencv_yaml(path: str, K: np.ndarray, D: np.ndarray, image_size: Tuple[int, int], meta: dict):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("D", D)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("rms", float(meta.get("rms", -1.0)))
    fs.write("global_rmse", float(meta.get("global_rmse", -1.0)))
    fs.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder or file. Supports images and videos.")
    ap.add_argument("--cols", type=int, required=True, help="Chessboard inner corners per row (columns). e.g., 9")
    ap.add_argument("--rows", type=int, required=True, help="Chessboard inner corners per column (rows). e.g., 6")
    ap.add_argument("--square", type=float, required=True, help="Square size in meters (or any unit). e.g., 0.025")
    ap.add_argument("--sample_every", type=int, default=10, help="For video: sample every N frames.")
    ap.add_argument("--max_video_frames", type=int, default=300, help="Max sampled frames per video to try.")
    ap.add_argument("--min_sharpness", type=float, default=50.0, help="Filter blurry frames. Lower to accept more.")
    ap.add_argument("--min_coverage", type=float, default=0.02, help="Filter tiny board in image. 0~1.")
    ap.add_argument("--use_rational", action="store_true", help="Enable k4,k5,k6. Useful for some lenses.")
    ap.add_argument("--reject_outliers", action="store_true", help="Reject high-error frames then recalibrate.")
    ap.add_argument("--outlier_sigma", type=float, default=2.0, help="Outlier threshold = mean + sigma*std")
    ap.add_argument("--save_used", action="store_true", help="Save visualization of used detections.")
    ap.add_argument("--out_dir", default="calib_out", help="Output directory.")
    args = ap.parse_args()

    media = list_media_files(args.input)
    if not media:
        raise RuntimeError("No media files found.")

    os.makedirs(args.out_dir, exist_ok=True)
    used_dir = os.path.join(args.out_dir, "used_frames") if args.save_used else None

    detections: List[DetectedFrame] = []

    for p in media:
        if is_image_file(p):
            d = extract_from_image(p, args.cols, args.rows, used_dir)
            if d is None:
                continue
            if d.sharpness < args.min_sharpness:
                continue
            if d.coverage < args.min_coverage:
                continue
            detections.append(d)

        elif is_video_file(p):
            ds = extract_from_video(
                p, args.cols, args.rows,
                sample_every_n_frames=args.sample_every,
                max_frames_to_process=args.max_video_frames,
                save_used_dir=used_dir
            )
            for d in ds:
                if d.sharpness < args.min_sharpness:
                    continue
                if d.coverage < args.min_coverage:
                    continue
                detections.append(d)

    if len(detections) < 8:
        raise RuntimeError(f"Valid detections too few: {len(detections)}. "
                           f"Try lowering --min_sharpness/--min_coverage or provide more images.")

    # Sort by coverage then sharpness (help stability)
    detections.sort(key=lambda x: (x.coverage, x.sharpness), reverse=True)

    # Optional: keep top-N to avoid too many near-duplicate frames
    # You can adjust if you have lots of frames.
    if len(detections) > 120:
        detections = detections[:120]

    result = calibrate_intrinsics(
        detections=detections,
        board_cols=args.cols,
        board_rows=args.rows,
        square_size=args.square,
        use_rational_model=args.use_rational,
        do_outlier_rejection=args.reject_outliers,
        outlier_sigma=args.outlier_sigma
    )

    K = result["K"]
    D = result["D"]
    w, h = result["image_size"]
    kept = result["kept_indices"]

    # Build report
    report = {
        "input": args.input,
        "board": {"cols": args.cols, "rows": args.rows, "square": args.square},
        "image_size": {"width": w, "height": h},
        "used_total": len(detections),
        "used_kept": len(kept),
        "use_rational_model": bool(args.use_rational),
        "reject_outliers": bool(args.reject_outliers),
        "rms": result["rms"],
        "global_rmse": result["global_rmse"],
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "per_view_rmse": result["per_view_rmse"],
        "frames": [
            {
                "source": detections[i].source_path,
                "frame_index": detections[i].frame_index,
                "sharpness": detections[i].sharpness,
                "coverage": detections[i].coverage,
                "saved_path": detections[i].saved_path
            }
            for i in kept
        ]
    }

    yaml_path = os.path.join(args.out_dir, "camera_intrinsics.yaml")
    json_path = os.path.join(args.out_dir, "calib_report.json")

    write_opencv_yaml(yaml_path, K, D, (w, h), {"rms": result["rms"], "global_rmse": result["global_rmse"]})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Calibration done ===")
    print(f"Kept frames: {len(kept)} / {len(detections)}")
    print(f"RMS (opencv): {result['rms']:.6f}")
    print(f"Global reprojection RMSE (px): {result['global_rmse']:.6f}")
    print(f"Output YAML: {yaml_path}")
    print(f"Output JSON: {json_path}")
    print("K=\n", K)
    print("D=\n", D.reshape(-1))


if __name__ == "__main__":
    main()

