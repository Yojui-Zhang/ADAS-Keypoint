#include "system_config_validation.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "system_config.h"

namespace {

bool Fail(const std::string& message, std::string* out_error) {
  if (out_error != nullptr) {
    *out_error = message;
  }
  return false;
}

bool IsFiniteNonnegative(const float value) {
  return std::isfinite(value) && value >= 0.0f;
}

bool ValidateThrottleConfig(const controller::ThrottleRuntimeConfig& cfg,
                            std::string* out_error) {
  if (cfg.calibration_id.empty()) {
    return Fail("throttle.calibration_id must not be empty", out_error);
  }

  if (cfg.control_period_ms < 5 || cfg.control_period_ms > 100) {
    return Fail("throttle.control_period_ms must be within [5, 100]",
                out_error);
  }

  if (cfg.vehicle_speed_timeout_ms <
          static_cast<std::uint64_t>(cfg.control_period_ms) ||
      cfg.acceleration_timeout_ms <
          static_cast<std::uint64_t>(cfg.control_period_ms)) {
    return Fail("throttle sensor timeouts must be at least one control period",
                out_error);
  }

  if (!std::isfinite(cfg.brake_interlock_threshold_0_10) ||
      cfg.brake_interlock_threshold_0_10 < 0.0f ||
      cfg.brake_interlock_threshold_0_10 > 1.0f) {
    return Fail(
        "throttle.brake_interlock_threshold_0_10 must be within [0, 1]",
        out_error);
  }

  const auto& pid = cfg.speed_pid;

  if (!IsFiniteNonnegative(pid.pedal_min_v) ||
      !IsFiniteNonnegative(pid.pedal_hard_max_v) ||
      pid.pedal_hard_max_v <= pid.pedal_min_v ||
      pid.pedal_hard_max_v > 5.0f) {
    return Fail("invalid throttle.speed_pid pedal voltage limits", out_error);
  }

  if (!std::isfinite(pid.integral_min_v) ||
      !std::isfinite(pid.integral_max_v) ||
      pid.integral_max_v < pid.integral_min_v) {
    return Fail("invalid throttle.speed_pid integral limits", out_error);
  }

  if (!IsFiniteNonnegative(pid.pedal_rise_rate_v_per_s) ||
      !IsFiniteNonnegative(pid.pedal_fall_rate_v_per_s) ||
      !IsFiniteNonnegative(pid.coast_fall_rate_v_per_s) ||
      !IsFiniteNonnegative(pid.coast_integral_decay_per_s)) {
    return Fail("invalid throttle.speed_pid transition rates", out_error);
  }

  if (!std::isfinite(pid.actual_speed_profile_weight) ||
      !std::isfinite(pid.reference_speed_profile_weight) ||
      pid.actual_speed_profile_weight < 0.0f ||
      pid.reference_speed_profile_weight < 0.0f ||
      pid.actual_speed_profile_weight + pid.reference_speed_profile_weight <=
          0.0f) {
    return Fail("throttle speed profile weights must have a positive sum",
                out_error);
  }

  if (!std::isfinite(pid.min_dt_s) ||
      !std::isfinite(pid.max_dt_s) ||
      pid.min_dt_s <= 0.0f ||
      pid.max_dt_s < pid.min_dt_s) {
    return Fail("invalid throttle.speed_pid dt limits", out_error);
  }

  const float period_s =
      static_cast<float>(cfg.control_period_ms) / 1000.0f;
  if (period_s < pid.min_dt_s || period_s > pid.max_dt_s) {
    return Fail("throttle control period is outside speed PID dt limits",
                out_error);
  }

  if (pid.profiles.size() < 2U) {
    return Fail("throttle.speed_pid.profiles requires at least two entries",
                out_error);
  }

  float previous_speed_kmh = -1.0f;
  for (std::size_t index = 0; index < pid.profiles.size(); ++index) {
    const controller::SpeedPedalProfile& profile = pid.profiles[index];

    if (!IsFiniteNonnegative(profile.speed_kmh) ||
        profile.speed_kmh <= previous_speed_kmh) {
      std::ostringstream oss;
      oss << "throttle profile speeds must be strictly increasing at index "
          << index;
      return Fail(oss.str(), out_error);
    }

    if (!std::isfinite(profile.feedforward_pedal_v) ||
        profile.feedforward_pedal_v < pid.pedal_min_v ||
        profile.feedforward_pedal_v > pid.pedal_hard_max_v) {
      return Fail(
          "throttle profile feedforward voltage is outside hard limits",
          out_error);
    }

    if (!std::isfinite(profile.pedal_upper_v) ||
        profile.pedal_upper_v < profile.feedforward_pedal_v ||
        profile.pedal_upper_v > pid.pedal_hard_max_v) {
      return Fail("throttle profile upper voltage is invalid", out_error);
    }

    if (!IsFiniteNonnegative(profile.kp_v_per_kmh) ||
        !IsFiniteNonnegative(profile.ki_v_per_kmh_s) ||
        !std::isfinite(profile.max_visible_error_kmh) ||
        profile.max_visible_error_kmh <= 0.0f) {
      return Fail("invalid throttle profile PI parameters", out_error);
    }

    previous_speed_kmh = profile.speed_kmh;
  }

  const auto& transition = cfg.mode_transition;
  if (!IsFiniteNonnegative(transition.coast_entry_delay_s) ||
      transition.coast_integral_retention < 0.0f ||
      transition.coast_integral_retention > 1.0f ||
      transition.speed_hold_integral_retention < 0.0f ||
      transition.speed_hold_integral_retention > 1.0f) {
    return Fail("invalid throttle mode transition configuration", out_error);
  }

  const auto& safety = cfg.safety;
  if (!std::isfinite(safety.acceleration_filter_tau_s) ||
      safety.acceleration_filter_tau_s <= 0.0f ||
      safety.low_speed_accel_limit_mps2 <= 0.0f ||
      safety.high_speed_accel_limit_mps2 <= 0.0f ||
      safety.high_speed_transition_kmh <= 0.0f ||
      safety.maximum_positive_jerk_mps3 <= 0.0f ||
      safety.hard_acceleration_margin_mps2 < 0.0f ||
      safety.hard_release_rate_v_per_s <= 0.0f) {
    return Fail("invalid throttle longitudinal safety configuration",
                out_error);
  }

  return true;
}

bool ValidateAccClosingGuard(const acc::AccConfig& cfg,
                             std::string* out_error) {
  const auto& guard = cfg.long_range_closing_guard;

  if (!guard.enabled) {
    return true;
  }

  if (!std::isfinite(guard.minimum_guard_distance_m) ||
      guard.minimum_guard_distance_m < 0.0f ||
      guard.minimum_guard_distance_m > cfg.max_forward_m) {
    return Fail(
        "ACC closing guard distance must be inside ACC detection range",
        out_error);
  }

  const auto valid_score = [](const float score) {
    return std::isfinite(score) && score >= 0.0f && score <= 1.0f;
  };

  if (!valid_score(guard.minimum_track_score) ||
      !valid_score(guard.strong_minimum_track_score)) {
    return Fail("ACC closing guard scores must be within [0, 1]",
                out_error);
  }

  if (!IsFiniteNonnegative(guard.maximum_rel_speed_std_mps) ||
      !IsFiniteNonnegative(guard.strong_maximum_rel_speed_std_mps) ||
      !IsFiniteNonnegative(guard.confirm_time_s) ||
      !IsFiniteNonnegative(guard.strong_confirm_time_s) ||
      !IsFiniteNonnegative(guard.strong_closing_speed_mps) ||
      !IsFiniteNonnegative(guard.unconfirmed_closing_cap_mps)) {
    return Fail(
        "ACC closing guard parameters must be finite and nonnegative",
        out_error);
  }

  if (guard.strong_confirm_time_s > guard.confirm_time_s) {
    return Fail(
        "strong closing confirmation must not be slower than normal confirmation",
        out_error);
  }

  if (guard.unconfirmed_closing_cap_mps >= guard.strong_closing_speed_mps) {
    return Fail(
        "unconfirmed closing cap must be below strong closing threshold",
        out_error);
  }

  if (cfg.high_speed_brake_time_gap_s > cfg.high_speed_coast_time_gap_s) {
    return Fail("high-speed brake time gap must not exceed coast time gap",
                out_error);
  }

  return true;
}

}  // namespace

bool ValidateSystemConfig(const AdasSystemConfig& config,
                          std::string* out_error) {
  if (!ValidateThrottleConfig(config.throttle, out_error)) {
    return false;
  }

  if (!ValidateAccClosingGuard(config.acc, out_error)) {
    return false;
  }

  return true;
}
