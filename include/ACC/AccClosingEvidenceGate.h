#pragma once

#include <algorithm>
#include <cmath>

namespace acc {

struct LongRangeClosingGuardConfig {
  bool enabled = true;

  float minimum_guard_distance_m = 30.0f;

  float minimum_track_score = 0.45f;
  float maximum_rel_speed_std_mps = 1.50f;
  float confirm_time_s = 0.25f;

  float strong_closing_speed_mps = 6.0f;
  float strong_minimum_track_score = 0.60f;
  float strong_maximum_rel_speed_std_mps = 1.00f;
  float strong_confirm_time_s = 0.10f;

  float unconfirmed_closing_cap_mps = 0.50f;
};

struct ClosingEvidenceInput {
  bool has_lead = false;
  int lead_id = -1;

  float distance_m = 0.0f;
  float raw_relative_speed_mps = 0.0f;
  float relative_speed_std_mps = 0.0f;
  float track_score = 0.0f;

  bool relative_speed_valid = false;
  float dt_s = 0.05f;
};

struct ClosingEvidenceOutput {
  float control_relative_speed_mps = 0.0f;

  bool guard_active = false;
  bool closing_confirmed = false;
  bool measurement_credible = false;

  float confirm_time_s = 0.0f;
};

class LongRangeClosingEvidenceGate {
public:
  explicit LongRangeClosingEvidenceGate(
      LongRangeClosingGuardConfig config = {})
      : config_(config) {}

  void SetConfig(const LongRangeClosingGuardConfig& config) noexcept {
    config_ = config;
  }

  ClosingEvidenceOutput Update(const ClosingEvidenceInput& input) noexcept {
    ClosingEvidenceOutput output;

    if (!input.has_lead ||
        input.lead_id < 0 ||
        !std::isfinite(input.distance_m) ||
        !std::isfinite(input.raw_relative_speed_mps)) {
      Reset();
      return output;
    }

    if (input.lead_id != active_lead_id_) {
      active_lead_id_ = input.lead_id;
      confirm_time_s_ = 0.0f;
    }

    const float dt_s =
        std::clamp(std::isfinite(input.dt_s) ? input.dt_s : 0.05f,
                   0.005f,
                   0.20f);

    const float raw_relative_speed_mps = input.raw_relative_speed_mps;

    if (raw_relative_speed_mps >= 0.0f) {
      confirm_time_s_ = std::max(0.0f, confirm_time_s_ - 2.0f * dt_s);

      output.control_relative_speed_mps = raw_relative_speed_mps;
      output.closing_confirmed = true;
      output.confirm_time_s = confirm_time_s_;
      return output;
    }

    const float closing_speed_mps = -raw_relative_speed_mps;

    output.guard_active =
        config_.enabled &&
        input.distance_m >=
            std::max(0.0f, config_.minimum_guard_distance_m);

    const bool normal_measurement_credible =
        input.relative_speed_valid &&
        std::isfinite(input.track_score) &&
        input.track_score >= std::max(0.0f, config_.minimum_track_score) &&
        std::isfinite(input.relative_speed_std_mps) &&
        input.relative_speed_std_mps <=
            std::max(0.0f, config_.maximum_rel_speed_std_mps);

    const bool strong_measurement_credible =
        input.relative_speed_valid &&
        closing_speed_mps >=
            std::max(0.0f, config_.strong_closing_speed_mps) &&
        std::isfinite(input.track_score) &&
        input.track_score >=
            std::max(0.0f, config_.strong_minimum_track_score) &&
        std::isfinite(input.relative_speed_std_mps) &&
        input.relative_speed_std_mps <=
            std::max(0.0f, config_.strong_maximum_rel_speed_std_mps);

    output.measurement_credible =
        normal_measurement_credible || strong_measurement_credible;

    if (output.measurement_credible) {
      confirm_time_s_ += dt_s;
    } else {
      confirm_time_s_ = std::max(0.0f, confirm_time_s_ - 2.0f * dt_s);
    }

    const float required_confirm_time_s =
        strong_measurement_credible
            ? std::max(0.0f, config_.strong_confirm_time_s)
            : std::max(0.0f, config_.confirm_time_s);

    output.closing_confirmed =
        !output.guard_active ||
        (output.measurement_credible &&
         confirm_time_s_ >= required_confirm_time_s);

    const float control_closing_speed_mps =
        output.closing_confirmed
            ? closing_speed_mps
            : std::min(closing_speed_mps,
                       std::max(0.0f,
                                config_.unconfirmed_closing_cap_mps));

    output.control_relative_speed_mps = -control_closing_speed_mps;
    output.confirm_time_s = confirm_time_s_;
    return output;
  }

  void Reset() noexcept {
    active_lead_id_ = -1;
    confirm_time_s_ = 0.0f;
  }

private:
  LongRangeClosingGuardConfig config_{};
  int active_lead_id_ = -1;
  float confirm_time_s_ = 0.0f;
};

}  // namespace acc
