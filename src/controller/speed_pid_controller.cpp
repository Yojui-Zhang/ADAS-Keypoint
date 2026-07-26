#include "speed_pid_controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace controller {

std::vector<SpeedPedalProfile> MakeDefaultSpeedPedalProfiles() {
  return {
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
  };
}

namespace {

constexpr float kAbsoluteMaxVisibleErrorKmh = 10.0f;

float Lerp(float lower, float upper, float ratio) noexcept {
  return lower + (upper - lower) * ratio;
}

const std::vector<SpeedPedalProfile>& DefaultProfiles() {
  static const std::vector<SpeedPedalProfile> profiles =
      MakeDefaultSpeedPedalProfiles();
  return profiles;
}

SpeedPedalProfile InterpolateProfile(
    const std::vector<SpeedPedalProfile>& profiles,
    float target_speed_kmh) {
  if (!std::isfinite(target_speed_kmh)) {
    throw std::invalid_argument("target_speed_kmh must be finite");
  }
  if (profiles.empty()) {
    throw std::invalid_argument("speed pedal profiles must not be empty");
  }

  const float speed_kmh = std::max(0.0f, target_speed_kmh);
  if (speed_kmh <= profiles.front().speed_kmh) {
    return profiles.front();
  }
  if (speed_kmh >= profiles.back().speed_kmh) {
    return profiles.back();
  }

  for (std::size_t index = 1; index < profiles.size(); ++index) {
    const SpeedPedalProfile& upper = profiles[index];
    if (speed_kmh > upper.speed_kmh) {
      continue;
    }

    const SpeedPedalProfile& lower = profiles[index - 1];
    const float speed_span = upper.speed_kmh - lower.speed_kmh;
    if (speed_span <= 0.0f) {
      throw std::invalid_argument("speed pedal profile speeds must increase");
    }
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

  return profiles.back();
}

}  // namespace

SpeedPedalController::SpeedPedalController(SpeedPedalControllerConfig config)
    : config_(config) {
  const auto finite_nonnegative = [](const float value) {
    return std::isfinite(value) && value >= 0.0f;
  };

  if (!finite_nonnegative(config_.pedal_min_v) ||
      !finite_nonnegative(config_.pedal_hard_max_v) ||
      config_.pedal_hard_max_v <= config_.pedal_min_v) {
    throw std::invalid_argument("invalid pedal voltage limits");
  }

  if (!finite_nonnegative(config_.speed_error_deadband_kmh) ||
      !std::isfinite(config_.integral_min_v) ||
      !std::isfinite(config_.integral_max_v) ||
      config_.integral_max_v < config_.integral_min_v) {
    throw std::invalid_argument("invalid speed PI limits");
  }

  if (!finite_nonnegative(config_.pedal_rise_rate_v_per_s) ||
      !finite_nonnegative(config_.pedal_fall_rate_v_per_s) ||
      !finite_nonnegative(config_.coast_fall_rate_v_per_s) ||
      !finite_nonnegative(config_.coast_integral_decay_per_s)) {
    throw std::invalid_argument("invalid pedal transition rates");
  }

  if (!finite_nonnegative(config_.min_dt_s) ||
      !finite_nonnegative(config_.max_dt_s) ||
      config_.min_dt_s <= 0.0f ||
      config_.max_dt_s < config_.min_dt_s) {
    throw std::invalid_argument("invalid controller time limits");
  }

  if (!std::isfinite(config_.actual_speed_profile_weight) ||
      !std::isfinite(config_.reference_speed_profile_weight) ||
      config_.actual_speed_profile_weight < 0.0f ||
      config_.reference_speed_profile_weight < 0.0f ||
      config_.actual_speed_profile_weight +
              config_.reference_speed_profile_weight <=
          0.0f) {
    throw std::invalid_argument("invalid speed profile weights");
  }

  if (config_.profiles.size() < 2U) {
    throw std::invalid_argument("speed pedal profiles require at least two entries");
  }

  float previous_speed_kmh = -1.0f;
  for (const SpeedPedalProfile& profile : config_.profiles) {
    if (!finite_nonnegative(profile.speed_kmh) ||
        profile.speed_kmh <= previous_speed_kmh ||
        !std::isfinite(profile.feedforward_pedal_v) ||
        profile.feedforward_pedal_v < config_.pedal_min_v ||
        profile.feedforward_pedal_v > config_.pedal_hard_max_v ||
        !std::isfinite(profile.pedal_upper_v) ||
        profile.pedal_upper_v < profile.feedforward_pedal_v ||
        profile.pedal_upper_v > config_.pedal_hard_max_v ||
        !finite_nonnegative(profile.kp_v_per_kmh) ||
        !finite_nonnegative(profile.ki_v_per_kmh_s) ||
        !std::isfinite(profile.max_visible_error_kmh) ||
        profile.max_visible_error_kmh <= 0.0f) {
      throw std::invalid_argument("invalid speed pedal profile");
    }

    previous_speed_kmh = profile.speed_kmh;
  }

  Reset();
}

float SpeedPedalController::Compute(float target_speed_kmh,
                                    float actual_speed_kmh,
                                    float dt_s) {
  if (!std::isfinite(target_speed_kmh) ||
      !std::isfinite(actual_speed_kmh) ||
      !std::isfinite(dt_s)) {
    ForceIdle();
    telemetry_.target_speed_kmh =
        std::isfinite(target_speed_kmh) ? std::max(0.0f, target_speed_kmh) : 0.0f;
    telemetry_.actual_speed_kmh =
        std::isfinite(actual_speed_kmh) ? std::max(0.0f, actual_speed_kmh) : 0.0f;
    return config_.pedal_min_v;
  }

  const float valid_dt_s = Clamp(dt_s, config_.min_dt_s, config_.max_dt_s);
  const float valid_target_speed_kmh = std::max(0.0f, target_speed_kmh);
  const float valid_actual_speed_kmh = std::max(0.0f, actual_speed_kmh);
  const float visible_target_kmh =
      LimitTargetSpeed(valid_target_speed_kmh, valid_actual_speed_kmh);
  const float profile_weight_sum =
      config_.actual_speed_profile_weight +
      config_.reference_speed_profile_weight;
  const float operating_speed_kmh =
      (config_.actual_speed_profile_weight * valid_actual_speed_kmh +
       config_.reference_speed_profile_weight * visible_target_kmh) /
      profile_weight_sum;
  const SpeedPedalProfile profile = SelectProfile(operating_speed_kmh);

  float speed_error_kmh = visible_target_kmh - valid_actual_speed_kmh;
  if (std::fabs(speed_error_kmh) <= config_.speed_error_deadband_kmh) {
    speed_error_kmh = 0.0f;
  }

  const float proportional_v = profile.kp_v_per_kmh * speed_error_kmh;
  const float integral_candidate_v =
      Clamp(integral_v_ + profile.ki_v_per_kmh_s * speed_error_kmh * valid_dt_s,
            config_.integral_min_v,
            config_.integral_max_v);

  const float pedal_upper_v = profile.pedal_upper_v;
  const float unsaturated_candidate_v =
      profile.feedforward_pedal_v + proportional_v + integral_candidate_v;

  const float bounded_candidate_v =
      Clamp(unsaturated_candidate_v, config_.pedal_min_v, pedal_upper_v);
  const float maximum_rise_v = config_.pedal_rise_rate_v_per_s * valid_dt_s;
  const float maximum_fall_v = config_.pedal_fall_rate_v_per_s * valid_dt_s;
  const float reachable_lower_v =
      std::max(config_.pedal_min_v, last_output_v_ - maximum_fall_v);
  const float reachable_upper_v =
      std::min(config_.pedal_hard_max_v, last_output_v_ + maximum_rise_v);

  const bool pushing_unreachable_high =
      speed_error_kmh > 0.0f &&
      (unsaturated_candidate_v > pedal_upper_v ||
       bounded_candidate_v > reachable_upper_v);
  const bool pushing_unreachable_low =
      speed_error_kmh < 0.0f &&
      (unsaturated_candidate_v < config_.pedal_min_v ||
       bounded_candidate_v < reachable_lower_v);

  if (!pushing_unreachable_high && !pushing_unreachable_low) {
    integral_v_ = integral_candidate_v;
  }

  const float desired_pedal_v =
      Clamp(profile.feedforward_pedal_v + proportional_v + integral_v_,
            config_.pedal_min_v,
            pedal_upper_v);

  last_output_v_ = SlewToward(desired_pedal_v,
                              config_.pedal_rise_rate_v_per_s,
                              config_.pedal_fall_rate_v_per_s,
                              valid_dt_s);
  last_output_v_ = Clamp(last_output_v_,
                         config_.pedal_min_v,
                         config_.pedal_hard_max_v);

  telemetry_.target_speed_kmh = valid_target_speed_kmh;
  telemetry_.actual_speed_kmh = valid_actual_speed_kmh;
  telemetry_.visible_target_speed_kmh = visible_target_kmh;
  telemetry_.operating_speed_kmh = operating_speed_kmh;
  telemetry_.speed_error_kmh = speed_error_kmh;
  telemetry_.feedforward_pedal_v = profile.feedforward_pedal_v;
  telemetry_.proportional_v = proportional_v;
  telemetry_.integral_v = integral_v_;
  telemetry_.desired_pedal_v = desired_pedal_v;
  telemetry_.final_pedal_v = last_output_v_;
  telemetry_.pedal_upper_v = pedal_upper_v;

  return last_output_v_;
}

float SpeedPedalController::ReleaseToIdle(float dt_s) noexcept {
  if (!std::isfinite(dt_s)) {
    ForceIdle();
    return config_.pedal_min_v;
  }

  const float valid_dt_s = Clamp(dt_s, config_.min_dt_s, config_.max_dt_s);
  const float integral_retention =
      Clamp(1.0f - config_.coast_integral_decay_per_s * valid_dt_s,
            0.0f,
            1.0f);

  integral_v_ = Clamp(integral_v_ * integral_retention,
                      config_.integral_min_v,
                      config_.integral_max_v);

  last_output_v_ = SlewToward(config_.pedal_min_v,
                              0.0f,
                              config_.coast_fall_rate_v_per_s,
                              valid_dt_s);
  last_output_v_ = Clamp(last_output_v_,
                         config_.pedal_min_v,
                         config_.pedal_hard_max_v);

  telemetry_.visible_target_speed_kmh = telemetry_.actual_speed_kmh;
  telemetry_.speed_error_kmh = 0.0f;
  telemetry_.feedforward_pedal_v = config_.pedal_min_v;
  telemetry_.proportional_v = 0.0f;
  telemetry_.integral_v = integral_v_;
  telemetry_.desired_pedal_v = config_.pedal_min_v;
  telemetry_.final_pedal_v = last_output_v_;
  telemetry_.pedal_upper_v = config_.pedal_hard_max_v;

  return last_output_v_;
}

void SpeedPedalController::ForceIdle() noexcept {
  integral_v_ = 0.0f;
  last_output_v_ = config_.pedal_min_v;
  telemetry_ = SpeedPedalControllerTelemetry{};
  telemetry_.feedforward_pedal_v = config_.pedal_min_v;
  telemetry_.desired_pedal_v = config_.pedal_min_v;
  telemetry_.final_pedal_v = config_.pedal_min_v;
  telemetry_.pedal_upper_v = config_.pedal_hard_max_v;
}

void SpeedPedalController::Reset() noexcept {
  ForceIdle();
}

void SpeedPedalController::PrepareForCoast(float integral_retention) noexcept {
  const float retention = Clamp(integral_retention, 0.0f, 1.0f);
  integral_v_ = Clamp(integral_v_ * retention,
                      config_.integral_min_v,
                      config_.integral_max_v);

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

void SpeedPedalController::SynchronizeAppliedOutput(
    const float applied_pedal_v) noexcept {
  if (!std::isfinite(applied_pedal_v)) {
    ForceIdle();
    return;
  }

  last_output_v_ = Clamp(applied_pedal_v,
                         config_.pedal_min_v,
                         config_.pedal_hard_max_v);
  telemetry_.final_pedal_v = last_output_v_;
}

float SpeedPedalController::LastOutputV() const noexcept {
  return last_output_v_;
}

SpeedPedalControllerTelemetry SpeedPedalController::LastTelemetry() const noexcept {
  return telemetry_;
}

float SpeedPedalController::SlewToward(float desired_pedal_v,
                                       float rise_rate_v_per_s,
                                       float fall_rate_v_per_s,
                                       float dt_s) const noexcept {
  const float maximum_rise_v =
      std::max(0.0f, rise_rate_v_per_s) * std::max(0.0f, dt_s);
  const float maximum_fall_v =
      std::max(0.0f, fall_rate_v_per_s) * std::max(0.0f, dt_s);

  return Clamp(desired_pedal_v,
               last_output_v_ - maximum_fall_v,
               last_output_v_ + maximum_rise_v);
}

float SpeedPedalController::Clamp(float value, float lower, float upper) noexcept {
  return std::max(lower, std::min(value, upper));
}

SpeedPedalProfile SpeedPedalController::SelectProfile(
    float operating_speed_kmh) const {
  return InterpolateProfile(config_.profiles, operating_speed_kmh);
}

float SpeedPedalController::LimitTargetSpeed(float desired_speed_kmh,
                                             float current_speed_kmh) const {
  if (!std::isfinite(desired_speed_kmh) || !std::isfinite(current_speed_kmh)) {
    return 0.0f;
  }

  const float valid_desired_speed_kmh = std::max(0.0f, desired_speed_kmh);
  const float valid_current_speed_kmh = std::max(0.0f, current_speed_kmh);
  const float speed_error_kmh =
      valid_desired_speed_kmh - valid_current_speed_kmh;
  if (speed_error_kmh <= 0.0f) {
    return valid_desired_speed_kmh;
  }

  const float preliminary_reference_kmh =
      valid_current_speed_kmh +
      std::min(speed_error_kmh, kAbsoluteMaxVisibleErrorKmh);
  const float profile_weight_sum =
      config_.actual_speed_profile_weight +
      config_.reference_speed_profile_weight;
  const float operating_speed_kmh =
      (config_.actual_speed_profile_weight * valid_current_speed_kmh +
       config_.reference_speed_profile_weight * preliminary_reference_kmh) /
      profile_weight_sum;
  const SpeedPedalProfile profile = SelectProfile(operating_speed_kmh);

  return valid_current_speed_kmh +
         std::min(speed_error_kmh, profile.max_visible_error_kmh);
}

SpeedPedalProfile SelectSpeedPedalProfile(float operating_speed_kmh) {
  return InterpolateProfile(DefaultProfiles(), operating_speed_kmh);
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

  const float preliminary_reference_kmh =
      valid_current_speed_kmh +
      std::min(speed_error_kmh, kAbsoluteMaxVisibleErrorKmh);
  const float operating_speed_kmh =
      0.75f * valid_current_speed_kmh + 0.25f * preliminary_reference_kmh;
  const SpeedPedalProfile profile =
      SelectSpeedPedalProfile(operating_speed_kmh);

  return valid_current_speed_kmh +
         std::min(speed_error_kmh, profile.max_visible_error_kmh);
}

}  // namespace controller
