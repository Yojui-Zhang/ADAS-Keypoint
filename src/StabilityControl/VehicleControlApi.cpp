#include "VehicleControlApi.h"
#include <mutex>
#include <limits>

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
  acc::ACC_SetEgoSpeedKmh(ego_speed_mps * 3.6f);
  const acc::AccCommand acc_cmd = acc::ACC_Run(world_result);

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
