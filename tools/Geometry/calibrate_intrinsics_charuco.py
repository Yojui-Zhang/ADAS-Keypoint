import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass
class CharucoDetection:
    source_path: str
    frame_index: int
    image_size: Tuple[int, int]  # (w,h)
    charuco_corners: np.ndarray  # (M,1,2) float32
    charuco_ids: np.ndarray      # (M,1) int32
    sharpness: float
    saved_path: Optional[str] = None


def is_image_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


def is_video_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in VID_EXTS


def list_media_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    out = []
    for root, _, names in os.walk(input_path):
        for n in names:
            p = os.path.join(root, n)
            if is_image_file(p) or is_video_file(p):
                out.append(p)
    out.sort()
    return out


def compute_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_aruco_dictionary(dict_name: str):
    # Common choices: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_APRILTAG_36h11 (if available)
    name = dict_name.upper()
    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown dictionary: {dict_name}. Example: DICT_4X4_50")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_charuco(
    img_bgr: np.ndarray,
    board,
    aruco_dict,
    detector_params,
    min_corners: int,
    do_subpix: bool = True
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
    if ids is None or len(ids) == 0:
        return None

    # Optional: refine detected markers (helps in difficult cases)
    try:
        cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
    except Exception:
        pass

    # Interpolate charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    if charuco_ids is None or charuco_corners is None:
        return None
    if int(retval) < min_corners:
        return None

    # Sub-pixel refinement on charuco corners
    if do_subpix:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-6)
        cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), criteria)

    return charuco_corners.astype(np.float32), charuco_ids.astype(np.int32)


def extract_from_image(path, board, aruco_dict, detector_params, min_corners, save_used_dir):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    det = detect_charuco(img, board, aruco_dict, detector_params, min_corners)
    if det is None:
        return None
    charuco_corners, charuco_ids = det
    sharp = compute_sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    saved = None
    if save_used_dir:
        os.makedirs(save_used_dir, exist_ok=True)
        vis = img.copy()
        # Draw detected markers and charuco corners
        corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), aruco_dict, parameters=detector_params)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
        base = os.path.splitext(os.path.basename(path))[0]
        saved = os.path.join(save_used_dir, f"{base}_used.png")
        cv2.imwrite(saved, vis)

    return CharucoDetection(path, -1, (w, h), charuco_corners, charuco_ids, sharp, saved)


def extract_from_video(path, board, aruco_dict, detector_params, min_corners,
                       sample_every, max_frames_to_process, save_used_dir):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    used = []
    idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_every != 0:
            idx += 1
            continue

        processed += 1
        if processed > max_frames_to_process:
            break

        h, w = frame.shape[:2]
        det = detect_charuco(frame, board, aruco_dict, detector_params, min_corners)
        if det is None:
            idx += 1
            continue

        charuco_corners, charuco_ids = det
        sharp = compute_sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        saved = None
        if save_used_dir:
            os.makedirs(save_used_dir, exist_ok=True)
            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
            base = os.path.splitext(os.path.basename(path))[0]
            saved = os.path.join(save_used_dir, f"{base}_frame{idx:06d}_used.png")
            cv2.imwrite(saved, vis)

        used.append(CharucoDetection(path, idx, (w, h), charuco_corners, charuco_ids, sharp, saved))
        idx += 1

    cap.release()
    return used


def reprojection_errors_charuco(all_corners, all_ids, rvecs, tvecs, K, D, board):
    per_view = []
    total_sq = 0.0
    total_n = 0

    for i in range(len(all_corners)):
        objp, imgp = board.matchImagePoints(all_corners[i], all_ids[i])
        if objp is None or imgp is None or len(objp) < 4:
            per_view.append(float("inf"))
            continue

        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        obs = imgp.reshape(-1, 2)
        err = np.linalg.norm(proj - obs, axis=1)

        rmse = float(np.sqrt(np.mean(err**2)))
        per_view.append(rmse)

        total_sq += float(np.sum(err**2))
        total_n += err.shape[0]

    global_rmse = float(np.sqrt(total_sq / max(1, total_n)))
    return per_view, global_rmse


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
    ap.add_argument("--dict", default="DICT_4X4_50", help="Aruco dictionary, e.g. DICT_4X4_50")
    ap.add_argument("--squares_x", type=int, required=True, help="Number of chessboard squares in X direction (NOT corners).")
    ap.add_argument("--squares_y", type=int, required=True, help="Number of chessboard squares in Y direction (NOT corners).")
    ap.add_argument("--square_len", type=float, required=True, help="Chessboard square length in meters.")
    ap.add_argument("--marker_len", type=float, required=True, help="ArUco marker side length in meters.")
    ap.add_argument("--min_corners", type=int, default=20, help="Minimum charuco corners to accept a frame.")
    ap.add_argument("--sample_every", type=int, default=10, help="For video: sample every N frames.")
    ap.add_argument("--max_video_frames", type=int, default=300, help="Max sampled frames per video.")
    ap.add_argument("--min_sharpness", type=float, default=50.0, help="Filter blurry frames.")
    ap.add_argument("--use_rational", action="store_true", help="Enable k4,k5,k6.")
    ap.add_argument("--reject_outliers", action="store_true", help="Reject high-error frames then recalibrate.")
    ap.add_argument("--outlier_sigma", type=float, default=2.0)
    ap.add_argument("--save_used", action="store_true", help="Save visualization of used detections.")
    ap.add_argument("--out_dir", default="calib_out", help="Output directory.")
    args = ap.parse_args()

    media = list_media_files(args.input)
    if not media:
        raise RuntimeError("No media files found.")

    os.makedirs(args.out_dir, exist_ok=True)
    used_dir = os.path.join(args.out_dir, "used_frames") if args.save_used else None

    aruco_dict = build_aruco_dictionary(args.dict)
    detector_params = cv2.aruco.DetectorParameters()
    # You can tune parameters here if needed.

    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        float(args.square_len),
        float(args.marker_len),
        aruco_dict
    )

    detections: List[CharucoDetection] = []
    for p in media:
        if is_image_file(p):
            d = extract_from_image(p, board, aruco_dict, detector_params, args.min_corners, used_dir)
            if d is None:
                continue
            if d.sharpness < args.min_sharpness:
                continue
            detections.append(d)

        elif is_video_file(p):
            ds = extract_from_video(
                p, board, aruco_dict, detector_params, args.min_corners,
                sample_every=args.sample_every,
                max_frames_to_process=args.max_video_frames,
                save_used_dir=used_dir
            )
            for d in ds:
                if d.sharpness < args.min_sharpness:
                    continue
                detections.append(d)

    if len(detections) < 8:
        raise RuntimeError(f"Valid detections too few: {len(detections)}. Need >= 8, got {len(detections)}")

    # Ensure consistent image size
    w0, h0 = detections[0].image_size
    for d in detections:
        if d.image_size != (w0, h0):
            raise RuntimeError("Image sizes are not consistent. Use a single resolution.")

    # Prepare data for calibration
    all_corners = [d.charuco_corners for d in detections]
    all_ids = [d.charuco_ids for d in detections]

    flags = 0
    if args.use_rational:
        flags |= cv2.CALIB_RATIONAL_MODEL

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-12)

    # First pass
    rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=(w0, h0),
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria
    )

    per_view, global_rmse = reprojection_errors_charuco(all_corners, all_ids, rvecs, tvecs, K, D, board)
    kept_indices = list(range(len(detections)))

    # Outlier rejection + second pass
    if args.reject_outliers and len(detections) >= 12:
        mu = float(np.mean(per_view))
        sd = float(np.std(per_view) + 1e-9)
        thr = mu + args.outlier_sigma * sd

        new_kept = [i for i, e in enumerate(per_view) if e <= thr]
        if len(new_kept) >= 8 and len(new_kept) < len(detections):
            kept_indices = new_kept
            all_corners2 = [all_corners[i] for i in kept_indices]
            all_ids2 = [all_ids[i] for i in kept_indices]

            rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_corners2,
                charucoIds=all_ids2,
                board=board,
                imageSize=(w0, h0),
                cameraMatrix=None,
                distCoeffs=None,
                flags=flags,
                criteria=criteria
            )
            per_view, global_rmse = reprojection_errors_charuco(all_corners2, all_ids2, rvecs, tvecs, K, D, board)

    report = {
        "input": args.input,
        "charuco": {
            "dictionary": args.dict,
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_len": args.square_len,
            "marker_len": args.marker_len,
            "min_corners": args.min_corners
        },
        "image_size": {"width": w0, "height": h0},
        "used_total": len(detections),
        "used_kept": len(kept_indices),
        "use_rational_model": bool(args.use_rational),
        "reject_outliers": bool(args.reject_outliers),
        "rms": float(rms),
        "global_rmse": float(global_rmse),
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "per_view_rmse": per_view,
        "frames": [
            {
                "source": detections[i].source_path,
                "frame_index": detections[i].frame_index,
                "sharpness": detections[i].sharpness,
                "saved_path": detections[i].saved_path
            }
            for i in kept_indices
        ]
    }

    yaml_path = os.path.join(args.out_dir, "camera_intrinsics.yaml")
    json_path = os.path.join(args.out_dir, "calib_report.json")
    write_opencv_yaml(yaml_path, K, D, (w0, h0), {"rms": float(rms), "global_rmse": float(global_rmse)})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== ChArUco Calibration done ===")
    print(f"Kept frames: {len(kept_indices)} / {len(detections)}")
    print(f"RMS (opencv): {float(rms):.6f}")
    print(f"Global reprojection RMSE (px): {float(global_rmse):.6f}")
    print(f"Output YAML: {yaml_path}")
    print(f"Output JSON: {json_path}")
    print("K=\n", K)
    print("D=\n", D.reshape(-1))


if __name__ == "__main__":
    main()

