#include "control_target_selector.h"

#include <algorithm>
#include <cmath>

#include "AccApi.h"

namespace adas_app {

float SelectActuatorSpeedTargetKmh(const keypad::RuntimeControlState& control_state,
                                   const stability::VehicleControlCommand& cmd,
                                   float ego_speed_kmh) {
  if (control_state.longitudinal_controller != keypad::LongitudinalControllerKind::Pid) {
    return std::max(0.0f, cmd.speed_kmh);
  }

  const acc::AccCommand& acc_cmd = cmd.acc_cmd;
  if (cmd.brake_0_10 > 0.05f ||
      acc_cmd.longitudinal_phase == acc::AccLongitudinalPhase::Braking ||
      acc_cmd.longitudinal_phase == acc::AccLongitudinalPhase::Idle) {
    return 0.0f;
  }

  if (cmd.speed_kmh > 0.2f && cmd.speed_kmh + 2.0f < ego_speed_kmh) {
    return std::max(0.0f, cmd.speed_kmh);
  }

  if (acc_cmd.has_lead &&
      acc_cmd.lead_following_active &&
      std::isfinite(acc_cmd.TargetSpeedKmh) &&
      acc_cmd.TargetSpeedKmh > 0.2f &&
      acc_cmd.TargetSpeedKmh + 2.0f < ego_speed_kmh) {
    return std::max(0.0f, acc_cmd.TargetSpeedKmh);
  }

  if (acc_cmd.cruise_speed_kmh > 0.2f) {
    return acc_cmd.cruise_speed_kmh;
  }

  return std::max(0.0f, cmd.speed_kmh);
}

}  // namespace adas_app
