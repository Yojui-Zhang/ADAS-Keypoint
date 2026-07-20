#pragma once

#include <cstdint>

#include "runtime_control_state.h"

namespace controller {

enum class ThrottleControlMode : std::uint8_t {
  Disabled = 0,
  Coast = 1,
  SpeedHold = 2,
  SpeedTracking = 3,
};

struct ThrottleControlTelemetry {
  ThrottleControlMode mode = ThrottleControlMode::Disabled;
  ThrottleControlMode requested_mode = ThrottleControlMode::Disabled;
  ThrottleControlMode effective_mode = ThrottleControlMode::Disabled;
  float target_speed_kmh = 0.0f;
  float current_speed_kmh = 0.0f;
  float feedforward_pedal_v = 0.75f;
  float speed_error_kmh = 0.0f;
  float integral_v = 0.0f;
  float final_pedal_v = 0.75f;
  float pedal_upper_v = 3.45f;
  bool vehicle_speed_fresh = false;
  float vehicle_speed_age_ms = 0.0f;
  std::uint64_t vehicle_speed_timestamp_ns = 0;
};

inline const char* ThrottleControlModeName(ThrottleControlMode mode) noexcept {
  switch (mode) {
    case ThrottleControlMode::Disabled:
      return "disabled";
    case ThrottleControlMode::Coast:
      return "coast";
    case ThrottleControlMode::SpeedHold:
      return "speed_hold";
    case ThrottleControlMode::SpeedTracking:
      return "speed_tracking";
  }

  return "unknown";
}

void ApplyThrottleRuntime(bool active, LongitudinalControllerKind controller_kind);
void SetThrottleControlRequest(float target_speed_kmh, ThrottleControlMode mode);
ThrottleControlTelemetry GetThrottleControlTelemetry();
void StopThrottleRuntime();

}  // namespace controller
