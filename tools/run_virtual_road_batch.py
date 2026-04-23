#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
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


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


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


if __name__ == "__main__":
    main()
