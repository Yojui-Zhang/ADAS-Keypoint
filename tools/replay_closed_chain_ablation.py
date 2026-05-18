#!/usr/bin/env python3
"""Matched replay-to-virtual-road closed-chain ablation.

This tool consumes ADAS research logs produced from the same replay video under
different exported artifacts. It reconstructs nominal acceleration demand from
the logged raw LKA steer and ACC acceleration command, applies deterministic
guard variants, and evaluates the resulting command sequence in a shared
kinematic virtual-road simulation.

The experiment is offline and matched by frame index. It is not an on-road
closed-loop validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAS_ROOT = REPO_ROOT / "ADAS"
DEFAULT_OUT = REPO_ROOT / "Paper" / "analysis" / "generated"

DEFAULT_LOGS = {
    "deploy_fp32": REPO_ROOT
    / "數據/Replay-video/TFlite/Yolov8_pose_Deploy_Only-Baseline/FP32/research_drive_20260429_204435.csv",
    "deploy_int8": REPO_ROOT
    / "數據/Replay-video/TFlite/Yolov8_pose_Deploy_Only-Baseline/INT8/research_drive_20260430_000542.csv",
    "as_eat_fp32": REPO_ROOT
    / "數據/Replay-video/TFlite/Yolov8_pose_AS-EAT/FP32/research_drive_20260429_155905.csv",
    "as_eat_int8": REPO_ROOT
    / "數據/Replay-video/TFlite/Yolov8_pose_AS-EAT/INT8/research_drive_20260429_164537.csv",
}

ARTIFACT_LABELS = {
    "deploy_fp32": "Deploy-only FP32",
    "deploy_int8": "Deploy-only INT8",
    "as_eat_fp32": "AS-EAT FP32",
    "as_eat_int8": "AS-EAT INT8",
}

GUARD_LABELS = {
    "none": "None",
    "radial_only": "Radial-only",
    "comfort_only": "Lateral-comfort-only",
    "scalar_clip": "Scalar-clip",
    "jerk_only": "Jerk-only",
    "full": "Full guard",
}


@dataclass
class Config:
    wheelbase_m: float = 2.62
    steering_ratio: float = 2.5
    g: float = 9.81
    mu_static: float = 0.90
    mu_dynamic: float = 0.75
    mu_lowpass_alpha: float = 0.90
    lat_safety: float = 0.85
    total_safety: float = 0.90
    lat_accel_comfort_mps2: float = 2.5
    long_accel_comfort_mps2: float = 1.8
    long_decel_comfort_mps2: float = 2.8
    emergency_decel_cap_mps2: float = 6.0
    slip_enter_ratio: float = 0.98
    slip_exit_ratio: float = 0.85
    ttc_hard_guard_s: float = 0.8
    w_lat: float = 1.0
    w_long: float = 4.0
    max_jerk_acc_mps3: float = 2.0
    max_jerk_dec_mps3: float = 3.5
    alat_cmd_guard_ratio: float = 0.5
    fallback_ego_speed_kmh: float = 10.0
    simulation_speed_kmh: float = 10.0
    dt_s: float = 0.05
    lane_width_m: float = 3.76


@dataclass
class GuardState:
    in_slip: bool = False
    mu_eff: float = 0.85
    last_a_long_cmd_mps2: float = 0.0


def parse_number(value: str | None, default: float = math.nan) -> float:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"", "nan", "-nan"}:
        return default
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return default


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def window_local_endpoint_drift(
    path: list[tuple[float, float]],
    reference_path: list[tuple[float, float]],
    window_frames: int,
) -> tuple[float, float]:
    """Return local drift accumulated within fixed windows after re-origining.

    This avoids reporting long-horizon open-loop separation as a 5-s physical
    displacement. Each window compares the artifact displacement against the
    FP32-reference displacement over the same frame interval.
    """
    if (
        window_frames <= 1
        or len(path) < window_frames + 1
        or len(reference_path) < window_frames + 1
    ):
        return math.nan, math.nan
    n = min(len(path), len(reference_path))
    deltas: list[float] = []
    for i in range(0, n - window_frames, window_frames):
        j = i + window_frames
        dx = (path[j][0] - path[i][0]) - (reference_path[j][0] - reference_path[i][0])
        dy = (path[j][1] - path[i][1]) - (reference_path[j][1] - reference_path[i][1])
        deltas.append(math.hypot(dx, dy))
    if not deltas:
        return math.nan, math.nan
    return fmean(deltas), quantile(deltas, 0.95)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def read_simple_config(path: Path) -> Config:
    cfg = Config()
    if not path.exists():
        return cfg

    section = ""
    with path.open(newline="") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip() or line.startswith("%") or line.strip() == "---":
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1].strip()
                continue
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            value = value.strip().strip('"').strip("'")
            if not value:
                continue
            v = parse_number(value)
            if not math.isfinite(v):
                continue
            if section == "stability" and hasattr(cfg, key):
                setattr(cfg, key, v)
            elif section == "lka" and key == "wheel_base_m":
                cfg.wheelbase_m = v
            elif section == "lka" and key == "lane_width_m":
                cfg.lane_width_m = v
            elif section == "app" and key == "fallback_ego_speed_kmh":
                cfg.fallback_ego_speed_kmh = v
                cfg.simulation_speed_kmh = v
            elif section == "ablation" and key == "virtual_sim_dt_s":
                cfg.dt_s = v
            elif section == "ablation" and key == "virtual_road_lane_width_m":
                cfg.lane_width_m = v
    return cfg


def read_research_log(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as f:
        rows = {}
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(parse_number(row.get("frame_idx"), -1))
            if frame_idx >= 0:
                rows[frame_idx] = row
        return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_road_csv(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(";", ",")
            vals: list[float] = []
            for item in line.split(","):
                v = parse_number(item)
                if math.isfinite(v):
                    vals.append(v)
            if len(vals) >= 2:
                points.append((vals[0], vals[1]))
    if len(points) < 2:
        raise ValueError(f"virtual road has fewer than two points: {path}")
    return points


def build_reference_path(
    artifact_rows: dict[int, dict[str, str]],
    frame_ids: list[int],
    guard_variant: str,
    cfg: Config,
) -> list[tuple[float, float]]:
    guard_state = GuardState()
    x = 0.0
    y = 0.0
    heading = 0.0
    v_path = cfg.simulation_speed_kmh / 3.6
    path = [(x, y)]

    for frame_idx in frame_ids:
        row = artifact_rows[frame_idx]
        alat_nom, along_nom, ttc_s = nominal_from_log(row, cfg)
        alat_cmd, _along_cmd, _a_budget, _alat_strip = apply_guard(
            guard_variant, alat_nom, along_nom, ttc_s, cfg, guard_state
        )
        if v_path > 0.5:
            heading = wrap_pi(heading + (alat_cmd / max(v_path, 1e-6)) * cfg.dt_s)
        x += v_path * cfg.dt_s * math.cos(heading)
        y += v_path * cfg.dt_s * math.sin(heading)
        path.append((x, y))

    if len(path) < 2:
        raise RuntimeError("failed to build FP32 reference path")
    return path


def estimate_curvature(path: list[tuple[float, float]], i: int) -> float:
    if len(path) < 3:
        return 0.0
    i1 = min(max(i, 0), len(path) - 1)
    i0 = i1 - 1 if i1 > 0 else i1
    i2 = min(i1 + 1, len(path) - 1)
    if i0 == i1 or i1 == i2:
        return 0.0
    ax, ay = path[i1][0] - path[i0][0], path[i1][1] - path[i0][1]
    bx, by = path[i2][0] - path[i1][0], path[i2][1] - path[i1][1]
    cx, cy = path[i2][0] - path[i0][0], path[i2][1] - path[i0][1]
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    lc = math.hypot(cx, cy)
    if la < 1e-9 or lb < 1e-9 or lc < 1e-9:
        return 0.0
    return 2.0 * (ax * by - ay * bx) / (la * lb * lc)


def project_pose(
    x: float,
    y: float,
    heading: float,
    path: list[tuple[float, float]],
    *,
    center_idx: int | None = None,
    window: int = 0,
) -> tuple[bool, float, float, float]:
    best_d2 = math.inf
    best: tuple[bool, float, float, float] = (False, math.nan, math.nan, 0.0)
    if center_idx is None or window <= 0:
        start_i = 1
        end_i = len(path) - 1
    else:
        start_i = max(1, center_idx - window)
        end_i = min(len(path) - 1, center_idx + window)
    for i in range(start_i, end_i + 1):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-12:
            continue
        t = clamp(((x - x0) * dx + (y - y0) * dy) / seg2, 0.0, 1.0)
        qx, qy = x0 + dx * t, y0 + dy * t
        ex, ey = x - qx, y - qy
        d2 = ex * ex + ey * ey
        if d2 >= best_d2:
            continue
        best_d2 = d2
        seg = math.sqrt(seg2)
        cte = (dx * ey - dy * ex) / seg if seg > 1e-9 else 0.0
        ref_heading = math.atan2(dy, dx)
        heading_err = wrap_pi(heading - ref_heading)
        best = (True, cte, heading_err, estimate_curvature(path, i))
    return best


def project_to_feasible(
    alat: float, along: float, a_budget: float, alat_max: float, cfg: Config
) -> tuple[float, float]:
    x_alat = clamp(alat, -alat_max, alat_max)
    x_along = along
    r2 = x_alat * x_alat + x_along * x_along
    a2 = a_budget * a_budget
    if r2 <= a2:
        return x_alat, x_along

    along_max = math.sqrt(max(0.0, a2 - x_alat * x_alat))
    c1 = (x_alat, clamp(x_along, -along_max, along_max))

    r = math.sqrt(max(1e-12, r2))
    scale = a_budget / r
    c2 = (x_alat * scale, x_along * scale)

    def cost(c: tuple[float, float]) -> float:
        return cfg.w_lat * (c[0] - alat) ** 2 + cfg.w_long * (c[1] - along) ** 2

    return c1 if cost(c1) <= cost(c2) else c2


def limits_for(cfg: Config, state: GuardState, alat_nom: float) -> tuple[float, float]:
    alat_s_limit = cfg.mu_static * cfg.g
    if not state.in_slip:
        if abs(alat_nom) > alat_s_limit * cfg.slip_enter_ratio:
            state.in_slip = True
    else:
        if abs(alat_nom) < alat_s_limit * cfg.slip_exit_ratio:
            state.in_slip = False
    mu_target = cfg.mu_dynamic if state.in_slip else cfg.mu_static
    alpha = clamp(cfg.mu_lowpass_alpha, 0.0, 0.999)
    state.mu_eff = alpha * state.mu_eff + (1.0 - alpha) * mu_target
    a_budget = state.mu_eff * cfg.g * cfg.total_safety
    alat_strip = min(cfg.lat_accel_comfort_mps2, state.mu_eff * cfg.g * cfg.lat_safety)
    return a_budget, alat_strip


def apply_guard(
    variant: str,
    alat_nom: float,
    along_nom: float,
    ttc_s: float,
    cfg: Config,
    state: GuardState,
) -> tuple[float, float, float, float]:
    a_budget, alat_strip = limits_for(cfg, state, alat_nom)
    dt = cfg.dt_s

    if variant == "none":
        return alat_nom, along_nom, a_budget, alat_strip

    if variant == "radial_only":
        r = math.hypot(alat_nom, along_nom)
        if r <= a_budget:
            return alat_nom, along_nom, a_budget, alat_strip
        s = a_budget / max(r, 1e-12)
        return alat_nom * s, along_nom * s, a_budget, alat_strip

    if variant == "comfort_only":
        return clamp(alat_nom, -alat_strip, alat_strip), along_nom, a_budget, alat_strip

    if variant == "scalar_clip":
        alat = clamp(alat_nom, -alat_strip, alat_strip)
        along = clamp(
            along_nom,
            -cfg.long_decel_comfort_mps2,
            cfg.long_accel_comfort_mps2,
        )
        return alat, along, a_budget, alat_strip

    if variant == "jerk_only":
        lo = state.last_a_long_cmd_mps2 - cfg.max_jerk_dec_mps3 * dt
        hi = state.last_a_long_cmd_mps2 + cfg.max_jerk_acc_mps3 * dt
        along = clamp(along_nom, lo, hi)
        state.last_a_long_cmd_mps2 = along
        return alat_nom, along, a_budget, alat_strip

    if variant != "full":
        raise ValueError(f"unknown guard variant: {variant}")

    alat, along = project_to_feasible(alat_nom, along_nom, a_budget, alat_strip, cfg)

    alat_used = min(abs(alat_nom), a_budget)
    along_left = math.sqrt(max(0.0, a_budget * a_budget - alat_used * alat_used))
    along = clamp(along, -along_left, along_left)

    accel_allow = min(along_left, cfg.long_accel_comfort_mps2)
    decel_allow = min(along_left, cfg.long_decel_comfort_mps2)
    if math.isfinite(ttc_s) and ttc_s < cfg.ttc_hard_guard_s:
        decel_allow = min(along_left, min(cfg.emergency_decel_cap_mps2, a_budget))
    if along >= 0.0:
        along = min(along, accel_allow)
    else:
        along = max(along, -decel_allow)

    emergency = math.isfinite(ttc_s) and ttc_s < cfg.ttc_hard_guard_s
    if not emergency:
        lo = state.last_a_long_cmd_mps2 - cfg.max_jerk_dec_mps3 * dt
        hi = state.last_a_long_cmd_mps2 + cfg.max_jerk_acc_mps3 * dt
        along = clamp(along, lo, hi)
        alat, along = project_to_feasible(alat, along, a_budget, alat_strip, cfg)
        along = clamp(along, -along_left, along_left)

    state.last_a_long_cmd_mps2 = along
    return alat, along, a_budget, alat_strip


def nominal_from_log(row: dict[str, str], cfg: Config) -> tuple[float, float, float]:
    ego_speed_kmh = parse_number(row.get("ego_speed_kmh"), cfg.fallback_ego_speed_kmh)
    if not math.isfinite(ego_speed_kmh) or ego_speed_kmh < 0.1:
        ego_speed_kmh = cfg.fallback_ego_speed_kmh
    v = ego_speed_kmh / 3.6
    steer_deg = parse_number(row.get("lka_steer_deg_raw"), 0.0)
    if not math.isfinite(steer_deg):
        steer_deg = 0.0
    road_deg = steer_deg / max(1e-6, cfg.steering_ratio)
    kappa = math.tan(math.radians(road_deg)) / max(1e-6, cfg.wheelbase_m)
    alat = v * v * kappa

    along = parse_number(row.get("acc_control_accel_cmd_mps2"), 0.0)
    if not math.isfinite(along):
        along = 0.0
    ttc = parse_number(row.get("acc_target_ttc_s"), math.inf)
    return alat, along, ttc


def simulate(
    artifact_key: str,
    artifact_rows: dict[int, dict[str, str]],
    frame_ids: list[int],
    guard_variant: str,
    cfg: Config,
    road_path: list[tuple[float, float]],
    projection_window: int = 0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    guard_state = GuardState()
    x = 0.0
    y = 0.0
    heading = 0.0
    # The matched replay-to-simulation ablation uses a shared speed profile.
    # Longitudinal commands are evaluated for feasibility/jerk, but are not
    # integrated into the path speed; otherwise a CAN-disabled fixed-speed
    # replay would create an artificial speed trajectory.
    v_path = cfg.simulation_speed_kmh / 3.6
    last_along: float | None = None

    abs_cte: list[float] = []
    abs_heading: list[float] = []
    abs_jerk: list[float] = []
    detail_rows: list[dict[str, object]] = []
    lane_departures = 0
    friction_viol = 0
    lateral_comfort_viol_count = 0
    longitudinal_comfort_viol_count = 0
    comfort_viol = 0
    hard_feasible = 0
    strict_pass = 0
    rate_pass = 0
    valid_proj = 0
    max_abs_cte = 0.0
    max_norm_ratio = 0.0
    sim_path: list[tuple[float, float]] = [(x, y)]

    for step_i, frame_idx in enumerate(frame_ids, start=1):
        row = artifact_rows[frame_idx]
        alat_nom, along_nom, ttc_s = nominal_from_log(row, cfg)
        alat_cmd, along_cmd, a_budget, alat_strip = apply_guard(
            guard_variant, alat_nom, along_nom, ttc_s, cfg, guard_state
        )
        norm = math.hypot(alat_cmd, along_cmd)
        norm_ratio = norm / a_budget if a_budget > 1e-9 else math.inf
        max_norm_ratio = max(max_norm_ratio, norm_ratio)
        f_viol = norm > a_budget + 1e-9
        lateral_comfort_viol = abs(alat_cmd) > alat_strip + 1e-9
        longitudinal_comfort_viol = (
            along_cmd > cfg.long_accel_comfort_mps2 + 1e-9
            or along_cmd < -cfg.long_decel_comfort_mps2 - 1e-9
        )
        c_viol = lateral_comfort_viol or longitudinal_comfort_viol
        jerk_value = math.nan
        if f_viol:
            friction_viol += 1
        if lateral_comfort_viol:
            lateral_comfort_viol_count += 1
        if longitudinal_comfort_viol:
            longitudinal_comfort_viol_count += 1
        if c_viol:
            comfort_viol += 1

        if last_along is not None:
            jerk_value = abs((along_cmd - last_along) / cfg.dt_s)
            abs_jerk.append(jerk_value)
        jerk_viol = math.isfinite(jerk_value) and jerk_value > cfg.max_jerk_dec_mps3 + 1e-9
        last_along = along_cmd

        if not jerk_viol:
            rate_pass += 1

        if not f_viol and not lateral_comfort_viol:
            hard_feasible += 1
        if not f_viol and not c_viol and not jerk_viol:
            strict_pass += 1

        if v_path > 0.5:
            heading = wrap_pi(heading + (alat_cmd / max(v_path, 1e-6)) * cfg.dt_s)
        x += v_path * cfg.dt_s * math.cos(heading)
        y += v_path * cfg.dt_s * math.sin(heading)
        sim_path.append((x, y))

        proj_valid, cte, heading_err, road_kappa = project_pose(
            x,
            y,
            heading,
            road_path,
            center_idx=step_i,
            window=projection_window,
        )
        lane_departure = 0
        if proj_valid:
            valid_proj += 1
            abs_cte_val = abs(cte)
            abs_cte.append(abs_cte_val)
            abs_heading.append(abs(heading_err))
            max_abs_cte = max(max_abs_cte, abs_cte_val)
            lane_departure = int(abs_cte_val > cfg.lane_width_m * 0.5)
            lane_departures += lane_departure

        detail_rows.append(
            {
                "artifact": ARTIFACT_LABELS[artifact_key],
                "artifact_key": artifact_key,
                "guard": GUARD_LABELS[guard_variant],
                "guard_key": guard_variant,
                "frame_idx": frame_idx,
                "alat_nom_mps2": f"{alat_nom:.9f}",
                "along_nom_mps2": f"{along_nom:.9f}",
                "alat_cmd_mps2": f"{alat_cmd:.9f}",
                "along_cmd_mps2": f"{along_cmd:.9f}",
                "a_budget_mps2": f"{a_budget:.9f}",
                "alat_strip_mps2": f"{alat_strip:.9f}",
                "friction_violation": int(f_viol),
                "comfort_violation": int(c_viol),
                "lateral_comfort_violation": int(lateral_comfort_viol),
                "longitudinal_comfort_violation": int(longitudinal_comfort_viol),
                "jerk_violation": int(jerk_viol),
                "virtual_road_valid": int(proj_valid),
                "cte_m": f"{cte:.9f}" if proj_valid else "nan",
                "heading_err_rad": f"{heading_err:.9f}" if proj_valid else "nan",
                "road_kappa_m_inv": f"{road_kappa:.9f}" if proj_valid else "nan",
                "lane_departure": lane_departure,
                "sim_speed_mps": f"{v_path:.9f}",
            }
        )

    n = len(frame_ids)
    window5_mean, window5_p95 = window_local_endpoint_drift(
        sim_path,
        road_path,
        max(1, round(5.0 / cfg.dt_s)),
    )

    summary = {
        "artifact": ARTIFACT_LABELS[artifact_key],
        "artifact_key": artifact_key,
        "guard": GUARD_LABELS[guard_variant],
        "guard_key": guard_variant,
        "matched_frames": n,
        "valid_virtual_frames": valid_proj,
        "cte_mean_m": f"{fmean(abs_cte):.6f}" if abs_cte else "nan",
        "cte_p95_m": f"{quantile(abs_cte, 0.95):.6f}" if abs_cte else "nan",
        "cte_max_m": f"{max_abs_cte:.6f}" if abs_cte else "nan",
        "local_5s_drift_mean_m": f"{window5_mean:.6f}" if math.isfinite(window5_mean) else "nan",
        "local_5s_drift_p95_m": f"{window5_p95:.6f}" if math.isfinite(window5_p95) else "nan",
        "heading_p95_deg": f"{math.degrees(quantile(abs_heading, 0.95)):.6f}"
        if abs_heading
        else "nan",
        "lane_departure_pct": f"{100.0 * lane_departures / valid_proj:.6f}"
        if valid_proj
        else "nan",
        "friction_violation_pct": f"{100.0 * friction_viol / n:.6f}" if n else "nan",
        "lateral_comfort_pass_pct": f"{100.0 * (n - lateral_comfort_viol_count) / n:.6f}" if n else "nan",
        "longitudinal_comfort_pass_pct": f"{100.0 * (n - longitudinal_comfort_viol_count) / n:.6f}" if n else "nan",
        "rate_pass_pct": f"{100.0 * rate_pass / n:.6f}" if n else "nan",
        "comfort_violation_pct": f"{100.0 * comfort_viol / n:.6f}" if n else "nan",
        "hard_feasible_cmd_pct": f"{100.0 * hard_feasible / n:.6f}" if n else "nan",
        "strict_envelope_rate_pass_pct": f"{100.0 * strict_pass / n:.6f}" if n else "nan",
        "jerk_p95_mps3": f"{quantile(abs_jerk, 0.95):.6f}" if abs_jerk else "nan",
        "max_friction_ratio": f"{max_norm_ratio:.6f}",
    }
    return summary, detail_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ADAS_ROOT / "config/system_config.yaml")
    parser.add_argument("--road-csv", type=Path, default=ADAS_ROOT / "road_csv/arc.csv")
    parser.add_argument(
        "--reference-mode",
        choices=["deploy-fp32-none", "deploy-fp32-full", "csv"],
        default="deploy-fp32-none",
        help=(
            "Reference path for CTE: deploy-fp32-none uses the matched FP32 replay "
            "trajectory; csv uses --road-csv."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--projection-window",
        type=int,
        default=400,
        help=(
            "Local reference-path search half-window for long matched replays. "
            "Use 0 to scan the full reference path."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on matched frames after validity filtering; 0 keeps all frames.",
    )
    parser.add_argument("--require-lka-valid", action="store_true", default=True)
    for key, path in DEFAULT_LOGS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, default=path)
    args = parser.parse_args()

    logs = {
        key: read_research_log(getattr(args, key))
        for key in DEFAULT_LOGS
    }
    paths = {key: getattr(args, key) for key in DEFAULT_LOGS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing input logs: " + ", ".join(missing))

    frame_ids = sorted(set.intersection(*(set(rows) for rows in logs.values())))
    if args.require_lka_valid:
        frame_ids = [
            frame_idx
            for frame_idx in frame_ids
            if all(int(parse_number(logs[key][frame_idx].get("lka_reference_valid"), 0)) == 1 for key in logs)
        ]
    if not frame_ids:
        raise RuntimeError("no matched frames after validity filtering")
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]

    cfg = read_simple_config(args.config)
    if args.reference_mode == "csv":
        road_path = read_road_csv(args.road_csv)
    else:
        ref_guard = "full" if args.reference_mode == "deploy-fp32-full" else "none"
        road_path = build_reference_path(logs["deploy_fp32"], frame_ids, ref_guard, cfg)

    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for artifact_key in logs:
        for guard_key in GUARD_LABELS:
            summary, rows = simulate(
                artifact_key,
                logs[artifact_key],
                frame_ids,
                guard_key,
                cfg,
                road_path,
                projection_window=max(0, args.projection_window),
            )
            summaries.append(summary)
            details.extend(rows)

    summary_path = args.output_dir / "closed_chain_ablation_summary.csv"
    detail_path = args.output_dir / "closed_chain_ablation_detail.csv"
    evidence_path = args.output_dir / "closed_chain_ablation_evidence.txt"

    summary_fields = [
        "artifact",
        "artifact_key",
        "guard",
        "guard_key",
        "matched_frames",
        "valid_virtual_frames",
        "cte_mean_m",
        "cte_p95_m",
        "cte_max_m",
        "local_5s_drift_mean_m",
        "local_5s_drift_p95_m",
        "heading_p95_deg",
        "lane_departure_pct",
        "friction_violation_pct",
        "lateral_comfort_pass_pct",
        "longitudinal_comfort_pass_pct",
        "rate_pass_pct",
        "comfort_violation_pct",
        "hard_feasible_cmd_pct",
        "strict_envelope_rate_pass_pct",
        "jerk_p95_mps3",
        "max_friction_ratio",
    ]
    detail_fields = [
        "artifact",
        "artifact_key",
        "guard",
        "guard_key",
        "frame_idx",
        "alat_nom_mps2",
        "along_nom_mps2",
        "alat_cmd_mps2",
        "along_cmd_mps2",
        "a_budget_mps2",
        "alat_strip_mps2",
        "friction_violation",
        "comfort_violation",
        "lateral_comfort_violation",
        "longitudinal_comfort_violation",
        "jerk_violation",
        "virtual_road_valid",
        "cte_m",
        "heading_err_rad",
        "road_kappa_m_inv",
        "lane_departure",
        "sim_speed_mps",
    ]
    write_csv(summary_path, summary_fields, summaries)
    write_csv(detail_path, detail_fields, details)

    with evidence_path.open("w") as f:
        f.write("Matched replay-to-virtual-road closed-chain ablation\n")
        f.write("Scope: offline matched replay-to-simulation; not on-road closed-loop validation.\n")
        f.write(f"Config: {args.config} sha256={sha256_file(args.config)}\n")
        f.write(f"Reference mode: {args.reference_mode}\n")
        f.write(f"Projection window: {max(0, args.projection_window)}\n")
        f.write(f"Road CSV: {args.road_csv} sha256={sha256_file(args.road_csv)}\n")
        f.write(f"Matched frames: {len(frame_ids)}\n")
        f.write(f"Frame range: {frame_ids[0]}..{frame_ids[-1]}\n")
        f.write(f"dt_s: {cfg.dt_s}\n")
        f.write(f"lane_width_m: {cfg.lane_width_m}\n")
        for key, path in paths.items():
            f.write(f"{key}: {path} sha256={sha256_file(path)} rows={len(logs[key])}\n")
        f.write(f"summary_csv: {summary_path}\n")
        f.write(f"detail_csv: {detail_path}\n")

    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {evidence_path}")


if __name__ == "__main__":
    main()
