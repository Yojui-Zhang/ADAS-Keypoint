#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit plane extrinsics for this ADAS project from ground-point CSV and "
            "write a Sensing-3M-style YAML. Project convention is raw world "
            "X=lateral_cm (positive=right), Y=forward_cm (positive=forward), Z=0."
        )
    )
    parser.add_argument(
        "--intrinsics-yaml",
        default="../Geometry-2025-v0/Geometry/calib_out/camera_intrinsics.yaml",
        help="OpenCV YAML that contains K/D/image size/rms.",
    )
    parser.add_argument(
        "--ground-csv",
        default="Camera-Config/單應性1280x720.csv",
        help="Ground correspondence CSV.",
    )
    parser.add_argument(
        "--output-yaml",
        default="Camera-Config/Sensing-3M.yaml",
        help="Output YAML path.",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Only keep rows whose '解析度' column matches this value. Empty means keep all rows.",
    )
    parser.add_argument(
        "--depth-col",
        default="世界X_深度_cm",
        help="CSV column for forward distance in cm.",
    )
    parser.add_argument(
        "--lateral-col",
        default="世界Y_橫向_cm",
        help="CSV column for lateral distance in cm. Positive-right is expected by this project.",
    )
    parser.add_argument(
        "--u-col",
        default="像素X",
        help="CSV column for image u/x.",
    )
    parser.add_argument(
        "--v-col",
        default="像素Y",
        help="CSV column for image v/y.",
    )
    parser.add_argument(
        "--resolution-col",
        default="解析度",
        help="CSV column for resolution filter.",
    )
    parser.add_argument(
        "--invert-lateral-sign",
        action="store_true",
        help="Flip the lateral sign before fitting if your CSV uses positive-left instead of positive-right.",
    )
    return parser.parse_args()


def load_opencv_yaml(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"cannot open yaml: {path}")

    data = {
        "K": fs.getNode("K").mat(),
        "D": fs.getNode("D").mat(),
        "image_width": int(fs.getNode("image_width").real()),
        "image_height": int(fs.getNode("image_height").real()),
        "rms": float(fs.getNode("rms").real()),
        "global_rmse": float(fs.getNode("global_rmse").real()),
    }
    fs.release()

    if data["K"] is None or data["D"] is None:
        raise RuntimeError(f"K/D missing in yaml: {path}")
    return data


def load_ground_csv(args):
    csv_path = Path(args.ground_csv)
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.resolution and args.resolution_col in row:
                if row[args.resolution_col].strip() != args.resolution:
                    continue

            u = float(row[args.u_col])
            v = float(row[args.v_col])
            forward_cm = float(row[args.depth_col])
            lateral_cm = float(row[args.lateral_col])
            if args.invert_lateral_sign:
                lateral_cm = -lateral_cm

            rows.append(
                {
                    "row": row,
                    "u": u,
                    "v": v,
                    "forward_cm": forward_cm,
                    "lateral_cm": lateral_cm,
                }
            )

    if len(rows) < 4:
        raise RuntimeError("need at least 4 valid CSV rows")

    image_points = np.array([(r["u"], r["v"]) for r in rows], dtype=np.float64)
    object_points = np.array(
        [(r["lateral_cm"], r["forward_cm"], 0.0) for r in rows], dtype=np.float64
    )
    return rows, image_points, object_points


def homography_init(image_points, object_points, K, D):
    plane_points = object_points[:, :2]
    undist_norm = cv2.undistortPoints(image_points.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    H, _ = cv2.findHomography(plane_points, undist_norm, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")

    b1 = H[:, 0]
    b2 = H[:, 1]
    b3 = H[:, 2]

    if b3[2] < 0:
        b1 = -b1
        b2 = -b2
        b3 = -b3

    scale = 2.0 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    r1 = scale * b1
    r2 = scale * b2
    r3 = np.cross(r1, r2)
    t = (scale * b3).reshape(3, 1)

    R0 = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R0)
    R0 = U @ Vt
    if np.linalg.det(R0) < 0:
        U[:, -1] *= -1.0
        R0 = U @ Vt

    rvec0, _ = cv2.Rodrigues(R0)
    return rvec0, t


def refine_extrinsics(image_points, object_points, K, D, rvec0, tvec0):
    if hasattr(cv2, "solvePnPRefineLM"):
        return cv2.solvePnPRefineLM(object_points, image_points, K, D, rvec0, tvec0)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        D,
        rvec=rvec0,
        tvec=tvec0,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP refinement failed")
    return rvec, tvec


def compute_metrics(rows, image_points, object_points, K, D, R, tvec):
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    errors = np.linalg.norm(proj - image_points, axis=1)
    rmse = math.sqrt(np.mean(errors ** 2))
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    camera_center = -(R.T @ tvec).reshape(3)

    by_depth = {}
    for err, row in zip(errors, rows):
        by_depth.setdefault(int(row["forward_cm"]), []).append(float(err))

    return {
        "errors": errors,
        "rmse": rmse,
        "mean": mean_err,
        "max": max_err,
        "camera_center": camera_center,
        "by_depth": by_depth,
    }


def format_opencv_matrix(matrix):
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        rows, cols = 1, arr.shape[0]
    else:
        rows, cols = arr.shape

    flat = arr.reshape(-1)
    formatted = ", ".join(f"{x:.16g}" for x in flat)
    return (
        "!!opencv-matrix\n"
        f"   rows: {rows}\n"
        f"   cols: {cols}\n"
        "   dt: d\n"
        f"   data: [ {formatted} ]"
    )


def write_output_yaml(out_path, intrinsics, R, tvec, args, metrics):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = f"""%YAML:1.0
---
image_width: {intrinsics['image_width']}
image_height: {intrinsics['image_height']}
rms: {intrinsics['rms']:.17g}
global_rmse: {intrinsics['global_rmse']:.17g}

K: {format_opencv_matrix(intrinsics['K'])}

D: {format_opencv_matrix(intrinsics['D'])}


# Extrinsics (world -> camera): p_c = R_cw * p_w + t_cw
# Raw world axes used by this project are:
#   X = lateral_cm (positive = right)
#   Y = forward_cm (positive = forward)
#   Z = 0 on the ground plane
# Runtime conversion in GeometryFunction is:
#   x_forward_m = raw_Y * 0.01
#   y_left_m    = -raw_X * 0.01
# Source intrinsics: {args.intrinsics_yaml}
# Source ground CSV: {args.ground_csv}
# Source resolution filter: {args.resolution or 'ALL'}
# CSV lateral sign inverted: {1 if args.invert_lateral_sign else 0}
# Ground CSV reprojection RMSE (full set): {metrics['rmse']:.6f} px
R_cw: {format_opencv_matrix(R)}

t_cw: {format_opencv_matrix(tvec.reshape(3, 1))}
"""
    out_path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()

    intrinsics = load_opencv_yaml(args.intrinsics_yaml)
    rows, image_points, object_points = load_ground_csv(args)

    rvec0, tvec0 = homography_init(image_points, object_points, intrinsics["K"], intrinsics["D"])
    rvec, tvec = refine_extrinsics(
        image_points,
        object_points,
        intrinsics["K"],
        intrinsics["D"],
        rvec0,
        tvec0,
    )
    R, _ = cv2.Rodrigues(rvec)

    metrics = compute_metrics(rows, image_points, object_points, intrinsics["K"], intrinsics["D"], R, tvec)
    write_output_yaml(args.output_yaml, intrinsics, R, tvec, args, metrics)

    print(f"wrote: {args.output_yaml}")
    print(f"reprojection_rmse_px: {metrics['rmse']:.6f}")
    print(f"reprojection_mean_px: {metrics['mean']:.6f}")
    print(f"reprojection_max_px: {metrics['max']:.6f}")
    cx, cy, cz = metrics["camera_center"]
    print(f"camera_center_world_cm: [{cx:.6f}, {cy:.6f}, {cz:.6f}]")
    print("per_depth_mean_px:")
    for depth_cm in sorted(metrics["by_depth"]):
        values = np.array(metrics["by_depth"][depth_cm], dtype=np.float64)
        print(
            f"  {depth_cm:5d} cm -> mean {values.mean():.6f} px, "
            f"max {values.max():.6f} px"
        )


if __name__ == "__main__":
    main()
