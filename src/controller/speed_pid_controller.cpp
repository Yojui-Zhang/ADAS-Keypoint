#include "speed_pid_controller.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace controller {
namespace {

constexpr std::array<SpeedPedalProfile, 11> kSpeedPedalProfiles = {{
    {0.0f, 0.75f, 0.050f, 0.025f, 1.20f, 4.0f},
    {10.0f, 1.05f, 0.050f, 0.025f, 1.50f, 5.0f},
    {20.0f, 1.45f, 0.048f, 0.024f, 1.90f, 6.0f},
    {30.0f, 1.70f, 0.046f, 0.023f, 2.15f, 7.0f},
    {40.0f, 1.85f, 0.044f, 0.022f, 2.35f, 8.0f},
    {50.0f, 2.00f, 0.042f, 0.021f, 2.60f, 9.0f},
    {60.0f, 2.25f, 0.040f, 0.020f, 2.85f, 10.0f},
    {70.0f, 2.55f, 0.038f, 0.018f, 3.10f, 10.0f},
    {80.0f, 2.80f, 0.036f, 0.016f, 3.30f, 10.0f},
    {90.0f, 3.05f, 0.032f, 0.012f, 3.45f, 10.0f},
    {100.0f, 3.30f, 0.030f, 0.010f, 3.45f, 10.0f},
}};

float Lerp(float lower, float upper, float ratio) noexcept {
  return lower + (upper - lower) * ratio;
}

SpeedPedalProfile InterpolateProfile(float target_speed_kmh) {
  if (!std::isfinite(target_speed_kmh)) {
    throw std::invalid_argument("target_speed_kmh must be finite");
  }

  const float speed_kmh = std::max(0.0f, target_speed_kmh);
  if (speed_kmh <= kSpeedPedalProfiles.front().speed_kmh) {
    return kSpeedPedalProfiles.front();
  }
  if (speed_kmh >= kSpeedPedalProfiles.back().speed_kmh) {
    return kSpeedPedalProfiles.back();
  }

  for (std::size_t index = 1; index < kSpeedPedalProfiles.size(); ++index) {
    const SpeedPedalProfile& upper = kSpeedPedalProfiles[index];
    if (speed_kmh > upper.speed_kmh) {
      continue;
    }

    const SpeedPedalProfile& lower = kSpeedPedalProfiles[index - 1];
    const float speed_span = std::max(0.1f, upper.speed_kmh - lower.speed_kmh);
    const float ratio = (speed_kmh - lower.speed_kmh) / speed_span;

    SpeedPedalProfile output = lower;
    output.speed_kmh = speed_kmh;
    output.feedforward_pedal_v =
        Lerp(lower.feedforward_pedal_v, upper.feedforward_pedal_v, ratio);
    output.kp_v_per_kmh = Lerp(lower.kp_v_per_kmh, upper.kp_v_per_kmh, ratio);
    output.ki_v_per_kmh_s =
        Lerp(lower.ki_v_per_kmh_s, upper.ki_v_per_kmh_s, ratio);
    output.pedal_upper_v = Lerp(lower.pedal_upper_v, upper.pedal_upper_v, ratio);
    output.max_visible_error_kmh =
        Lerp(lower.max_visible_error_kmh, upper.max_visible_error_kmh, ratio);
    return output;
  }

  return kSpeedPedalProfiles.back();
}

}  // namespace

SpeedPedalController::SpeedPedalController(SpeedPedalControllerConfig config)
    : config_(config) {
  if (!std::isfinite(config_.pedal_min_v) ||
      !std::isfinite(config_.pedal_hard_max_v) ||
      config_.pedal_min_v < 0.0f ||
      config_.pedal_hard_max_v <= config_.pedal_min_v) {
    throw std::invalid_argument("invalid pedal voltage limits");
  }

  Reset();
}

float SpeedPedalController::Compute(float target_speed_kmh,
                                    float actual_speed_kmh,
                                    float dt_s) {
  if (!std::isfinite(target_speed_kmh) ||
      !std::isfinite(actual_speed_kmh) ||
      !std::isfinite(dt_s)) {
    Reset();
    telemetry_.target_speed_kmh =
        std::isfinite(target_speed_kmh) ? std::max(0.0f, target_speed_kmh) : 0.0f;
    telemetry_.actual_speed_kmh =
        std::isfinite(actual_speed_kmh) ? std::max(0.0f, actual_speed_kmh) : 0.0f;
    return config_.pedal_min_v;
  }

  const float valid_dt_s = Clamp(dt_s, config_.min_dt_s, config_.max_dt_s);
  const SpeedPedalProfile profile = SelectSpeedPedalProfile(target_speed_kmh);
  const float visible_target_kmh = LimitSpeedPidTarget(target_speed_kmh, actual_speed_kmh);

  float speed_error_kmh = visible_target_kmh - std::max(0.0f, actual_speed_kmh);
  if (std::fabs(speed_error_kmh) <= config_.speed_error_deadband_kmh) {
    speed_error_kmh = 0.0f;
  }

  const float proportional_v = profile.kp_v_per_kmh * speed_error_kmh;
  const float integral_candidate_v =
      Clamp(integral_v_ + profile.ki_v_per_kmh_s * speed_error_kmh * valid_dt_s,
            config_.integral_min_v,
            config_.integral_max_v);

  const float pedal_upper_v = std::min(profile.pedal_upper_v, config_.pedal_hard_max_v);
  const float unsaturated_candidate_v =
      profile.feedforward_pedal_v + proportional_v + integral_candidate_v;

  const bool saturated_high = unsaturated_candidate_v > pedal_upper_v;
  const bool saturated_low = unsaturated_candidate_v < config_.pedal_min_v;
  const bool can_integrate =
      (!saturated_high && !saturated_low) ||
      (saturated_high && speed_error_kmh < 0.0f) ||
      (saturated_low && speed_error_kmh > 0.0f);

  if (can_integrate) {
    integral_v_ = integral_candidate_v;
  }

  const float desired_pedal_v =
      Clamp(profile.feedforward_pedal_v + proportional_v + integral_v_,
            config_.pedal_min_v,
            pedal_upper_v);

  const float maximum_rise_v = config_.pedal_rise_rate_v_per_s * valid_dt_s;
  const float maximum_fall_v = config_.pedal_fall_rate_v_per_s * valid_dt_s;

  last_output_v_ = Clamp(desired_pedal_v,
                         last_output_v_ - maximum_fall_v,
                         last_output_v_ + maximum_rise_v);
  last_output_v_ = Clamp(last_output_v_, config_.pedal_min_v, pedal_upper_v);

  telemetry_.target_speed_kmh = std::max(0.0f, target_speed_kmh);
  telemetry_.actual_speed_kmh = std::max(0.0f, actual_speed_kmh);
  telemetry_.visible_target_speed_kmh = visible_target_kmh;
  telemetry_.speed_error_kmh = speed_error_kmh;
  telemetry_.feedforward_pedal_v = profile.feedforward_pedal_v;
  telemetry_.proportional_v = proportional_v;
  telemetry_.integral_v = integral_v_;
  telemetry_.final_pedal_v = last_output_v_;
  telemetry_.pedal_upper_v = pedal_upper_v;

  return last_output_v_;
}

void SpeedPedalController::Reset() noexcept {
  integral_v_ = 0.0f;
  last_output_v_ = config_.pedal_min_v;
  telemetry_ = SpeedPedalControllerTelemetry{};
  telemetry_.feedforward_pedal_v = config_.pedal_min_v;
  telemetry_.final_pedal_v = config_.pedal_min_v;
  telemetry_.pedal_upper_v = config_.pedal_hard_max_v;
}

void SpeedPedalController::PrepareForCoast(float integral_retention) noexcept {
  const float retention = Clamp(integral_retention, 0.0f, 1.0f);
  integral_v_ = Clamp(integral_v_ * retention,
                      config_.integral_min_v,
                      config_.integral_max_v);
  last_output_v_ = config_.pedal_min_v;

  telemetry_.integral_v = integral_v_;
  telemetry_.final_pedal_v = last_output_v_;
  telemetry_.pedal_upper_v = config_.pedal_hard_max_v;
}

void SpeedPedalController::PrepareForSpeedHold(float integral_retention) noexcept {
  const float retention = Clamp(integral_retention, 0.0f, 1.0f);
  integral_v_ = Clamp(integral_v_ * retention,
                      config_.integral_min_v,
                      config_.integral_max_v);

  telemetry_.integral_v = integral_v_;
  telemetry_.final_pedal_v = last_output_v_;
  telemetry_.pedal_upper_v = config_.pedal_hard_max_v;
}

float SpeedPedalController::LastOutputV() const noexcept {
  return last_output_v_;
}

SpeedPedalControllerTelemetry SpeedPedalController::LastTelemetry() const noexcept {
  return telemetry_;
}

float SpeedPedalController::Clamp(float value, float lower, float upper) noexcept {
  return std::max(lower, std::min(value, upper));
}

SpeedPedalProfile SelectSpeedPedalProfile(float target_speed_kmh) {
  return InterpolateProfile(target_speed_kmh);
}

float LimitSpeedPidTarget(float desired_speed_kmh, float current_speed_kmh) {
  if (!std::isfinite(desired_speed_kmh) || !std::isfinite(current_speed_kmh)) {
    return 0.0f;
  }

  const float valid_desired_speed_kmh = std::max(0.0f, desired_speed_kmh);
  const float valid_current_speed_kmh = std::max(0.0f, current_speed_kmh);
  const float speed_error_kmh = valid_desired_speed_kmh - valid_current_speed_kmh;
  if (speed_error_kmh <= 0.0f) {
    return valid_desired_speed_kmh;
  }

  const SpeedPedalProfile profile = SelectSpeedPedalProfile(valid_desired_speed_kmh);
  return valid_current_speed_kmh + std::min(speed_error_kmh, profile.max_visible_error_kmh);
}

}  // namespace controller
