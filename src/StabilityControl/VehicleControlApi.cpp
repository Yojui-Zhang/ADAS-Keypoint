#include "VehicleControlApi.h"

#include "GeometryAdapter.h"
#include "lk_centerline.h"
#include "lk_lane_points.h"

#include <cmath>
#include <mutex>
#include <limits>

namespace {

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
  std::lock_guard<std::mutex> lk(g_mtx);

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
  if (BuildAccScopedWorldResult(world_result, acc_world_result)) {
    acc_input = &acc_world_result;
  }

  acc::ACC_SetEgoSpeedKmh(ego_speed_mps * 3.6f);
  const acc::AccCommand acc_cmd = acc::ACC_Run(*acc_input);

  // 2) LKA
  std::string lka_dbg;
  const float steer_deg = lane_steering_step(world_result, ego_speed_mps, &lka_dbg);

  // 3) Supervisor projection
  VehicleControlCommand cmd = g_supervisor.Update(
      ego_speed_mps, dt_s,
      acc_cmd, steer_deg,
      yaw_rate, alat,
      "", lka_dbg
  );

  cmd.acc_cmd = acc_cmd;
  cmd.lka_steer_deg_raw = steer_deg;

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
