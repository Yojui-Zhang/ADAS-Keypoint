#include "control_target_selector.h"

#include <algorithm>
#include <cmath>

#include "AccApi.h"

namespace adas_app {

float SelectActuatorSpeedTargetKmh(const controller::RuntimeControlState& control_state,
                                   const stability::VehicleControlCommand& cmd,
                                   float ego_speed_kmh) {
  if (controller::DemoLongitudinalControlEnabled(control_state) == false) {
    return 0.0f;
  }

  const auto sanitize_speed = [](const float speed_kmh) {
    if (!std::isfinite(speed_kmh)) {
      return 0.0f;
    }
    return std::max(0.0f, speed_kmh);
  };

  if (control_state.longitudinal_controller != controller::LongitudinalControllerKind::Pid) {
    return sanitize_speed(cmd.speed_kmh);
  }

  const acc::AccCommand& acc_cmd = cmd.acc_cmd;
  if (cmd.brake_0_10 > 0.05f ||
      acc_cmd.longitudinal_phase == acc::AccLongitudinalPhase::Braking) {
    return 0.0f;
  }

  if (acc_cmd.longitudinal_phase == acc::AccLongitudinalPhase::Idle) {
    return 0.0f;
  }

  if (acc_cmd.longitudinal_phase == acc::AccLongitudinalPhase::Coasting) {
    return sanitize_speed(cmd.speed_kmh);
  }

  if (acc_cmd.stop_state == acc::AccStopState::Resuming) {
    return sanitize_speed(cmd.speed_kmh);
  }

  const float supervisor_speed_kmh = sanitize_speed(cmd.speed_kmh);
  if (supervisor_speed_kmh > 0.2f && supervisor_speed_kmh + 2.0f < ego_speed_kmh) {
    return supervisor_speed_kmh;
  }

  if (acc_cmd.has_lead &&
      acc_cmd.lead_following_active &&
      std::isfinite(acc_cmd.TargetSpeedKmh) &&
      acc_cmd.TargetSpeedKmh > 0.2f &&
      acc_cmd.TargetSpeedKmh + 2.0f < ego_speed_kmh) {
    return sanitize_speed(acc_cmd.TargetSpeedKmh);
  }

  if (acc_cmd.cruise_speed_kmh > 0.2f) {
    return sanitize_speed(acc_cmd.cruise_speed_kmh);
  }

  return supervisor_speed_kmh;
}

}  // namespace adas_app
