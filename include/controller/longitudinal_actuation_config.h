#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace controller {

struct SpeedPedalProfile {
  float speed_kmh = 0.0f;
  float feedforward_pedal_v = 0.75f;
  float kp_v_per_kmh = 0.0f;
  float ki_v_per_kmh_s = 0.0f;
  float pedal_upper_v = 3.45f;
  float max_visible_error_kmh = 10.0f;
};

std::vector<SpeedPedalProfile> MakeDefaultSpeedPedalProfiles();

struct SpeedPedalControllerConfig {
  float pedal_min_v = 0.75f;
  float pedal_hard_max_v = 3.45f;

  float speed_error_deadband_kmh = 0.30f;

  float integral_min_v = -0.40f;
  float integral_max_v = 0.35f;

  float pedal_rise_rate_v_per_s = 1.60f;
  float pedal_fall_rate_v_per_s = 3.00f;

  float coast_fall_rate_v_per_s = 3.00f;
  float coast_integral_decay_per_s = 2.50f;

  float actual_speed_profile_weight = 0.75f;
  float reference_speed_profile_weight = 0.25f;

  float min_dt_s = 0.005f;
  float max_dt_s = 0.100f;

  std::vector<SpeedPedalProfile> profiles = MakeDefaultSpeedPedalProfiles();
};

struct ThrottleModeTransitionConfig {
  float coast_entry_delay_s = 0.10f;
  float coast_integral_retention = 0.25f;
  float speed_hold_integral_retention = 0.15f;
};

struct LongitudinalPedalSafetyConfig {
  float acceleration_filter_tau_s = 0.15f;

  float low_speed_accel_limit_mps2 = 1.30f;
  float high_speed_accel_limit_mps2 = 1.05f;
  float high_speed_transition_kmh = 90.0f;

  float maximum_positive_jerk_mps3 = 1.80f;

  float hard_acceleration_margin_mps2 = 0.35f;
  float hard_release_rate_v_per_s = 4.00f;
};

struct ThrottleRuntimeConfig {
  std::string calibration_id = "throttle_default_v1";

  int control_period_ms = 20;

  std::uint64_t vehicle_speed_timeout_ms = 100;
  std::uint64_t acceleration_timeout_ms = 150;

  float brake_interlock_threshold_0_10 = 0.05f;

  SpeedPedalControllerConfig speed_pid{};
  ThrottleModeTransitionConfig mode_transition{};
  LongitudinalPedalSafetyConfig safety{};
};

}  // namespace controller
