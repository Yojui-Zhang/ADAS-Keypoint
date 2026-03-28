#pragma once
#include <string>
#include "StabilityConfig.h"

#include "AccApi.h"       // acc::AccCommand, ACC_GetConfig()
#include "lane_keeping.h" // lane_steering_step()

namespace stability {

struct VehicleControlPerfStats {
  double acc_scope_ms = 0.0;
  double acc_ms = 0.0;
  double lka_ms = 0.0;
  double stability_ms = 0.0;
  double total_ms = 0.0;
};

struct VehicleControlCommand {
  float steer_deg   = 0.0f;
  float speed_kmh   = 0.0f;
  float brake_0_10  = 0.0f;

  acc::AccCommand acc_cmd{};
  float lka_steer_deg_raw = 0.0f;
  VehicleControlPerfStats perf{};

  std::string debug;
};

class StabilitySupervisor {
public:
  explicit StabilitySupervisor(const StabilityConfig& cfg = {}) : cfg_(cfg) {}

  void SetConfig(const StabilityConfig& cfg) { cfg_ = cfg; }
  const StabilityConfig& GetConfig() const { return cfg_; }

  VehicleControlCommand Update(double ego_speed_mps,
                               double dt_s,
                               const acc::AccCommand& acc_cmd,
                               double lka_steer_deg,
                               double yaw_rate_rps,
                               double alat_mps2,
                               const std::string& acc_dbg,
                               const std::string& lka_dbg);

private:
  StabilityConfig cfg_;

  // slip state
  bool   in_slip_ = false;
  double mu_eff_  = 0.85;

  // outputs memory
  double last_speed_cmd_mps_    = 0.0;
  double last_steer_deg_        = 0.0;
  double last_a_long_cmd_mps2_  = 0.0;

  // IMU a_lat low-pass (abs)
  double alat_meas_lpf_mps2_ = 0.0;
  bool   alat_lpf_inited_    = false;
};

} // namespace stability
