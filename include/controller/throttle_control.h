#pragma once

#include <cstdint>
#include <string>

#include "longitudinal_actuation_config.h"
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
  float visible_target_speed_kmh = 0.0f;
  float operating_speed_kmh = 0.0f;
  float feedforward_pedal_v = 0.75f;
  float speed_error_kmh = 0.0f;
  float integral_v = 0.0f;
  float desired_pedal_v = 0.75f;
  float final_pedal_v = 0.75f;
  float applied_pedal_v = 0.75f;
  float pedal_upper_v = 3.45f;
  float requested_brake_0_10 = 0.0f;
  bool brake_interlock_active = false;
  float measured_dt_s = 0.0f;
  bool vehicle_speed_fresh = false;
  float vehicle_speed_age_ms = 0.0f;
  std::uint64_t vehicle_speed_timestamp_ns = 0;
  bool vehicle_acceleration_fresh = false;
  float raw_acceleration_mps2 = 0.0f;
  float filtered_acceleration_mps2 = 0.0f;
  float measured_jerk_mps3 = 0.0f;
  float allowed_acceleration_mps2 = 0.0f;
  bool acceleration_guard_active = false;
  bool jerk_guard_active = false;
  std::string calibration_id = "throttle_default_v1";
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
void ConfigureThrottleRuntime(const ThrottleRuntimeConfig& config);
void SetThrottleControlRequest(float target_speed_kmh,
                               ThrottleControlMode mode,
                               float brake_0_10);
ThrottleControlTelemetry GetThrottleControlTelemetry();
void StopThrottleRuntime();

}  // namespace controller
