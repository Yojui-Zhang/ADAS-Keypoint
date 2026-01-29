#include "VehicleControlApi.h"
#include <mutex>

namespace stability {

static std::mutex g_mtx;
static StabilityConfig g_stab_cfg;
static StabilitySupervisor g_supervisor(g_stab_cfg);

VehicleControlCommand VehicleControl_Run(const std::vector<TrackingBox>& world_result,
                                        float ego_speed_mps,
                                        float dt_s,
                                        std::string* out_debug)
{
  std::lock_guard<std::mutex> lk(g_mtx);

  // 1) ACC（餵入外部車速可以更穩） :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
  acc::ACC_SetEgoSpeedKmh(ego_speed_mps * 3.6f);
  const acc::AccCommand acc_cmd = acc::ACC_Run(world_result);

  // 2) LKA（讀取 steer） :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
  std::string lka_dbg;
  const float steer_deg = lane_steering_step(world_result, ego_speed_mps, &lka_dbg);

  // 3) Supervisor（摩擦/離心/動能約束，輸出最終命令）
  VehicleControlCommand cmd = g_supervisor.Update(
      static_cast<double>(ego_speed_mps),
      static_cast<double>(dt_s),
      acc_cmd,
      static_cast<double>(steer_deg),
      /*acc_dbg*/"",
      /*lka_dbg*/lka_dbg
  );

  if (out_debug) *out_debug = cmd.debug;
  
  cmd.acc_cmd = acc_cmd;
  cmd.lka_steer_deg_raw = steer_deg;
  return cmd;
}

void VehicleControl_SetStabilityConfig(const StabilityConfig& cfg) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_stab_cfg = cfg;
  g_supervisor.SetConfig(g_stab_cfg);
}

StabilityConfig VehicleControl_GetStabilityConfig() {
  std::lock_guard<std::mutex> lk(g_mtx);
  return g_stab_cfg;
}

} // namespace stability

