#include "VehicleControlApi.h"

#include "GeometryAdapter.h"
#include "lk_centerline.h"
#include "lk_lane_points.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <limits>
#include <sstream>

namespace {

using PerfClock = std::chrono::steady_clock;

double ElapsedMs(const PerfClock::time_point& start,
                 const PerfClock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool IsAccVehicleClass(const TrackingBox& tb)
{
  return tb.class_id == 1 || tb.class_id == 2 || tb.class_id == 3;
}

bool BuildAccScopedWorldResult(const std::vector<TrackingBox>& world_result,
                               std::vector<TrackingBox>& acc_world_result)
{
  const ControlConfig lka_cfg = lane_keeping_get_control_config();

  TrackingBox centerline;
  std::string centerline_debug;
  if (!lane_keeping::internal::BuildCenterlineFromWorldResult(
          world_result, lka_cfg, centerline, centerline_debug)) {
    return false;
  }

  std::vector<cv::Point2f> center_pts;
  center_pts.reserve(centerline.kpts.size());
  for (const auto& kp : centerline.kpts) {
    if (!std::isfinite(kp.x) || !std::isfinite(kp.y)) continue;
    center_pts.emplace_back(kp.x, kp.y);
  }
  if (center_pts.size() < 2) return false;

  const acc::AccConfig acc_cfg = acc::ACC_GetConfig();

  acc_world_result.clear();
  acc_world_result.reserve(world_result.size());

  for (const auto& tb : world_result) {
    if (!IsAccVehicleClass(tb)) {
      acc_world_result.push_back(tb);
      continue;
    }

    cv::Point2f ground_xy;
    if (!acc::TryGetGroundBottomCenterXY(tb, ground_xy)) continue;

    float lane_center_y_m = 0.0f;
    const bool has_lane_center =
        lane_keeping::internal::EstimateLaneYAtX(center_pts, ground_xy.x, lane_center_y_m);

    const float lane_relative_y_m =
        has_lane_center ? (ground_xy.y - lane_center_y_m) : ground_xy.y;

    if (std::fabs(lane_relative_y_m) <= acc_cfg.lateral_limit_m) {
      acc_world_result.push_back(tb);
    }
  }

  return true;
}

void DisableLongitudinalAccCommand(acc::AccCommand& cmd)
{
  cmd.speed_kmh = 0.0f;
  cmd.brake_0_10 = 0.0f;
  cmd.accel_cmd_mps2 = 0.0f;
  cmd.free_accel_nom_mps2 = 0.0f;
  cmd.free_accel_limited_mps2 = 0.0f;
  cmd.longitudinal_phase = acc::AccLongitudinalPhase::Idle;
}

stability::VehicleControlCommand BuildRawVehicleControlCommand(
    const acc::AccCommand& acc_cmd,
    float lka_steer_deg,
    const stability::VehicleControlOptions& options)
{
  stability::VehicleControlCommand cmd;
  cmd.steer_deg = options.enable_lateral_control ? lka_steer_deg : 0.0f;
  cmd.speed_kmh = options.enable_longitudinal_control
      ? std::max(0.0f, acc_cmd.speed_kmh)
      : 0.0f;
  cmd.brake_0_10 = options.enable_longitudinal_control
      ? std::clamp(acc_cmd.brake_0_10, 0.0f, 10.0f)
      : 0.0f;

  std::ostringstream oss;
  oss << "VehicleControl(RAW)"
      << " | lateral=" << (options.enable_lateral_control ? "on" : "off")
      << " longitudinal=" << (options.enable_longitudinal_control ? "on" : "off")
      << " supervisor=off"
      << " | steer=" << cmd.steer_deg
      << " | speed=" << cmd.speed_kmh
      << " | brake=" << cmd.brake_0_10;
  cmd.debug = oss.str();
  return cmd;
}

} // namespace

namespace stability {

static std::mutex g_mtx;
static StabilityConfig g_stab_cfg;
static StabilitySupervisor g_supervisor(g_stab_cfg);

static double g_yaw_rate_rps = std::numeric_limits<double>::quiet_NaN();
static double g_alat_mps2    = std::numeric_limits<double>::quiet_NaN();
static double g_imu_age_s    = 1e9;

void VehicleControl_SetImu(double yaw_rate_rps, double alat_mps2)
{
  std::lock_guard<std::mutex> lk(g_mtx);
  g_yaw_rate_rps = yaw_rate_rps;
  g_alat_mps2    = alat_mps2;
  g_imu_age_s    = 0.0;
}

VehicleControlCommand VehicleControl_Run(const std::vector<TrackingBox>& world_result,
                                        float ego_speed_mps,
                                        float dt_s,
                                        std::string* out_debug)
{
  const VehicleControlOptions options;
  return VehicleControl_RunWithOptions(world_result, ego_speed_mps, dt_s, options, out_debug);
}

VehicleControlCommand VehicleControl_RunWithOptions(const std::vector<TrackingBox>& world_result,
                                                    float ego_speed_mps,
                                                    float dt_s,
                                                    const VehicleControlOptions& options,
                                                    std::string* out_debug)
{
  std::lock_guard<std::mutex> lk(g_mtx);
  const auto vc_start = PerfClock::now();

  // age IMU
  g_imu_age_s += (dt_s > 0 ? dt_s : 0.0f);

  double yaw_rate = g_yaw_rate_rps;
  double alat     = g_alat_mps2;

  // timeout => treat as invalid (Supervisor will fallback to command-based bound)
  if (g_imu_age_s > g_stab_cfg.alat_meas_timeout_s) {
    yaw_rate = std::numeric_limits<double>::quiet_NaN();
    alat     = std::numeric_limits<double>::quiet_NaN();
  }

  // 1) ACC
  std::vector<TrackingBox> acc_world_result;
  const std::vector<TrackingBox>* acc_input = &world_result;
  const auto acc_scope_start = PerfClock::now();
  if (BuildAccScopedWorldResult(world_result, acc_world_result)) {
    acc_input = &acc_world_result;
  }
  const auto acc_scope_end = PerfClock::now();

  const auto acc_start = PerfClock::now();
  acc::ACC_SetEgoSpeedKmh(ego_speed_mps * 3.6f);
  const acc::AccCommand acc_cmd = acc::ACC_Run(*acc_input);
  const auto acc_end = PerfClock::now();

  // 2) LKA
  std::string lka_dbg;
  const auto lka_start = PerfClock::now();
  const float steer_deg = lane_steering_step(world_result, ego_speed_mps, &lka_dbg);
  const auto lka_end = PerfClock::now();

  acc::AccCommand control_acc_cmd = acc_cmd;
  if (options.enable_longitudinal_control == false) {
    DisableLongitudinalAccCommand(control_acc_cmd);
  }
  const float control_steer_deg =
      options.enable_lateral_control ? steer_deg : 0.0f;

  // 3) Supervisor projection or raw ACC/LKA command for demo ablation.
  const auto stability_start = PerfClock::now();
  VehicleControlCommand cmd;
  if (options.enable_supervisor) {
    cmd = g_supervisor.Update(
        ego_speed_mps, dt_s,
        control_acc_cmd, control_steer_deg,
        yaw_rate, alat,
        "", lka_dbg
    );
  } else {
    cmd = BuildRawVehicleControlCommand(control_acc_cmd, control_steer_deg, options);
  }
  if (options.enable_lateral_control == false) {
    cmd.steer_deg = 0.0f;
  }
  if (options.enable_longitudinal_control == false) {
    cmd.speed_kmh = 0.0f;
    cmd.brake_0_10 = 0.0f;
  }
  const auto stability_end = PerfClock::now();

  cmd.acc_cmd = acc_cmd;
  cmd.lka_steer_deg_raw = steer_deg;
  cmd.perf.acc_scope_ms = ElapsedMs(acc_scope_start, acc_scope_end);
  cmd.perf.acc_ms = ElapsedMs(acc_start, acc_end);
  cmd.perf.lka_ms = ElapsedMs(lka_start, lka_end);
  cmd.perf.stability_ms = ElapsedMs(stability_start, stability_end);
  cmd.perf.total_ms = ElapsedMs(vc_start, stability_end);

  if (out_debug) *out_debug = cmd.debug;
  return cmd;
}

void VehicleControl_SetStabilityConfig(const StabilityConfig& cfg)
{
  std::lock_guard<std::mutex> lk(g_mtx);
  g_stab_cfg = cfg;
  g_supervisor.SetConfig(g_stab_cfg);
}

StabilityConfig VehicleControl_GetStabilityConfig()
{
  std::lock_guard<std::mutex> lk(g_mtx);
  return g_stab_cfg;
}

} // namespace stability
