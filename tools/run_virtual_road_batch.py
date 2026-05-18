#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAS_ROOT = REPO_ROOT / "ADAS"
DEFAULT_BINARY = ADAS_ROOT / "tools" / "virtual_road_benchmark"
DEFAULT_OUTPUT = REPO_ROOT / "數據" / "CONTROLLER_V2"

SCENARIOS = [
    ("straight", ["--road-mode", "straight"]),
    ("arc_left_gentle", ["--road-mode", "arc", "--arc-radius-m", "300"]),
    ("arc_right_gentle", ["--road-mode", "arc", "--arc-radius-m", "-300"]),
    (
        "s_curve_gentle",
        ["--road-mode", "s_curve", "--s-amplitude-m", "1.0", "--s-wavelength-m", "120"],
    ),
]

SOURCE_FILES = [
    ADAS_ROOT / "tools" / "virtual_road_benchmark.cpp",
    ADAS_ROOT / "tools" / "run_virtual_road_batch.py",
    ADAS_ROOT / "src" / "log" / "algorithm_ablation_logger.cpp",
    ADAS_ROOT / "include" / "log" / "algorithm_ablation_logger.h",
]

SUMMARY_FIELDS = [
    "samples",
    "virtual_road_valid_ratio",
    "avg_abs_vc_cte_m",
    "max_abs_vc_cte_m",
    "avg_abs_preview_mpc_cte_m",
    "max_abs_preview_mpc_cte_m",
    "avg_abs_disturbed_preview_mpc_cte_m",
    "max_abs_disturbed_preview_mpc_cte_m",
    "avg_abs_raw_cte_m",
    "max_abs_raw_cte_m",
    "avg_abs_vc_heading_err_deg",
    "max_abs_vc_heading_err_deg",
    "avg_abs_preview_mpc_heading_err_deg",
    "max_abs_preview_mpc_heading_err_deg",
    "avg_abs_disturbed_preview_mpc_heading_err_deg",
    "max_abs_disturbed_preview_mpc_heading_err_deg",
    "avg_abs_raw_heading_err_deg",
    "max_abs_raw_heading_err_deg",
    "vc_lane_departure_ratio",
    "preview_mpc_lane_departure_ratio",
    "disturbed_preview_mpc_lane_departure_ratio",
    "raw_lane_departure_ratio",
    "virtual_sim_raw_steer_bias_deg",
    "virtual_sim_raw_steer_osc_amp_deg",
    "virtual_sim_raw_steer_osc_period_s",
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def write_comparison_csv(output_root: Path, rows: list[dict[str, str]]) -> Path:
    path = output_root / "disturbed_controller_comparison.csv"
    fieldnames = [
        "speed_kmh",
        "scenario",
        "source_csv",
        "source_summary",
    ] + SUMMARY_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_manifest(output_root: Path,
                   binary: Path,
                   rows: list[dict[str, str]],
                   comparison_csv: Path) -> Path:
    path = output_root / "RUN_MANIFEST.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"generated_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"command={' '.join(sys.argv)}\n")
        f.write(f"repo_root={REPO_ROOT}\n")
        f.write(f"binary={binary}\n")
        if binary.exists():
            f.write(f"binary_sha256={sha256_file(binary)}\n")
        f.write(f"comparison_csv={comparison_csv}\n")
        f.write("disturbed_preview_mpc_definition=preview-MPC control law plus the same deterministic steering disturbance injected into the raw baseline\n")
        f.write("\n[source_sha256]\n")
        for source in SOURCE_FILES:
            f.write(f"{source}={sha256_file(source)}\n")
        f.write("\n[outputs]\n")
        for row in rows:
            f.write(f"{row['speed_kmh']}km,{row['scenario']},{row['source_csv']},{row['source_summary']}\n")
    return path


def build_binary(binary: Path) -> None:
    src = ADAS_ROOT / "tools" / "virtual_road_benchmark.cpp"
    impl = ADAS_ROOT / "src" / "log" / "algorithm_ablation_logger.cpp"
    includes = [str(ADAS_ROOT), str(ADAS_ROOT / "include")]
    includes.extend(
        str(path)
        for path in sorted((ADAS_ROOT / "include").iterdir())
        if path.is_dir()
    )

    pkg_cflags = subprocess.check_output(["pkg-config", "--cflags", "opencv4"], text=True).strip().split()
    pkg_libs = subprocess.check_output(["pkg-config", "--libs", "opencv4"], text=True).strip().split()

    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-o",
        str(binary),
        str(src),
        str(impl),
    ]
    binary.parent.mkdir(parents=True, exist_ok=True)
    for inc in includes:
        cmd.extend(["-I", inc])
    cmd.extend(pkg_cflags)
    cmd.extend(pkg_libs)
    run(cmd)


def needs_rebuild(binary: Path) -> bool:
    if not binary.exists():
        return True
    newest_src = max(
        (ADAS_ROOT / "tools" / "virtual_road_benchmark.cpp").stat().st_mtime,
        (ADAS_ROOT / "src" / "log" / "algorithm_ablation_logger.cpp").stat().st_mtime,
        (ADAS_ROOT / "include" / "log" / "algorithm_ablation_logger.h").stat().st_mtime,
    )
    return binary.stat().st_mtime < newest_src


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate virtual-road controller baseline logs.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--dt-s", type=float, default=0.05)
    parser.add_argument("--lane-width-m", type=float, default=3.76)
    parser.add_argument("--max-steer-deg", type=float, default=35.0)
    parser.add_argument("--speeds-kmh", type=str, default="10,30,50,80")
    args = parser.parse_args()

    if shutil.which("pkg-config") is None:
        raise RuntimeError("pkg-config is required to build virtual_road_benchmark")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_rebuild(args.binary):
        build_binary(args.binary)

    frame_count = max(1, int(round(args.duration_s / args.dt_s)))
    speeds = [int(item.strip()) for item in args.speeds_kmh.split(",") if item.strip()]
    rows: list[dict[str, str]] = []

    for speed in speeds:
        speed_dir = args.output_root / f"{speed}km"
        speed_dir.mkdir(parents=True, exist_ok=True)
        for scenario_name, scenario_args in SCENARIOS:
            output_csv = speed_dir / f"ablation_drive_{scenario_name}_{speed}km.csv"
            cmd = [
                str(args.binary),
                "--output",
                str(output_csv),
                "--speed-kmh",
                str(speed),
                "--frames",
                str(frame_count),
                "--dt-s",
                str(args.dt_s),
                "--lane-width-m",
                str(args.lane_width_m),
                "--max-steer-deg",
                str(args.max_steer_deg),
            ]
            cmd.extend(scenario_args)
            run(cmd)
            print(f"[ok] speed={speed} scenario={scenario_name} -> {output_csv}")
            summary_path = Path(str(output_csv) + ".summary.txt")
            summary = read_summary(summary_path)
            row = {
                "speed_kmh": str(speed),
                "scenario": scenario_name,
                "source_csv": str(output_csv),
                "source_summary": str(summary_path),
            }
            row.update(summary)
            rows.append(row)

    comparison_csv = write_comparison_csv(args.output_root, rows)
    manifest_path = write_manifest(args.output_root, args.binary, rows, comparison_csv)
    print(f"[ok] comparison -> {comparison_csv}")
    print(f"[ok] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
