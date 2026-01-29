#pragma once
#include <string>
#include "StabilityConfig.h"

// 直接吃你現有 ACC/LKA 輸出
#include "AccApi.h"       // acc::AccCommand, ACC_GetConfig()
#include "lane_keeping.h" // lane_steering_step()

namespace stability {

struct VehicleControlCommand {
  float steer_deg = 0.0f;
  float speed_kmh = 0.0f;
  float brake_0_10 = 0.0f;

  // 把 ACC 原始輸出一起帶出來（用於畫框與 Target 資訊）
  acc::AccCommand acc_cmd{};

  // 可選：把 LKA 原始輸出也帶出來（方便 debug）
  float lka_steer_deg_raw = 0.0f;

  std::string debug;
};

class StabilitySupervisor {
public:
  explicit StabilitySupervisor(const StabilityConfig& cfg = {}) : cfg_(cfg) {}

  void SetConfig(const StabilityConfig& cfg) { cfg_ = cfg; }
  const StabilityConfig& GetConfig() const { return cfg_; }

  // 核心：以摩擦/離心/動能做安全包絡，對 ACC/LKA 命令「再限制」
  VehicleControlCommand Update(double ego_speed_mps,
                               double dt_s,
                               const acc::AccCommand& acc_cmd,
                               double lka_steer_deg,
                               const std::string& acc_dbg = "",
                               const std::string& lka_dbg = "");

private:
  StabilityConfig cfg_;

  // state
  bool   in_slip_ = false;
  double mu_eff_ = 0.85;

  double last_speed_cmd_mps_ = 0.0;
  double last_steer_deg_ = 0.0;
};

} // namespace stability

