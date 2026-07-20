#include "StopAndGoController.h"

#include <algorithm>
#include <cmath>

namespace acc {
namespace {

constexpr float kMpsToKmh = 3.6f;

}  // namespace

StopAndGoController::StopAndGoController(StopAndGoConfig config)
    : config_(config) {}

void StopAndGoController::SetConfig(const StopAndGoConfig& config) {
  config_ = config;

  if (!config_.enabled) {
    Reset();
  }
}

const StopAndGoConfig& StopAndGoController::GetConfig() const noexcept {
  return config_;
}

void StopAndGoController::Reset() noexcept {
  state_ = AccStopState::Moving;

  state_time_s_ = 0.0f;
  movement_confirm_time_s_ = 0.0f;
  target_lost_time_s_ = 0.0f;

  held_lead_id_ = -1;
  held_lead_distance_m_ = 0.0f;
  manual_resume_requested_ = false;
  resume_without_lead_active_ = false;
}

AccStopState StopAndGoController::State() const noexcept {
  return state_;
}

void StopAndGoController::RequestManualResume() noexcept {
  manual_resume_requested_ = true;
}

StopAndGoOutput StopAndGoController::Update(const StopAndGoInput& input) {
  StopAndGoOutput output{};
  output.state = state_;
  output.accel_cmd_mps2 = input.base_accel_mps2;

  if (!config_.enabled) {
    Reset();
    output.state = state_;
    return output;
  }

  manual_resume_requested_ = false;

  const float dt_s = Clamp(std::isfinite(input.dt_s) ? input.dt_s : 0.05f,
                           0.005f,
                           0.20f);

  state_time_s_ += dt_s;

  const bool valid_lead = IsLeadValid(input);
  const float ego_speed_kmh = std::max(0.0f, input.ego_speed_mps) * kMpsToKmh;
  const float lead_speed_kmh =
      std::max(0.0f, input.lead_speed_mps) * kMpsToKmh;
  const float standstill_gap_m = std::max(0.5f, input.standstill_gap_m);
  const float stop_trigger_distance_m =
      standstill_gap_m +
      std::max(0.0f, config_.stop_trigger_margin_m) +
      std::max(0.0f, input.ego_speed_mps) *
          std::max(0.0f, config_.stop_trigger_time_s);

  const bool low_speed_approach =
      ego_speed_kmh <=
      std::max(config_.hold_enter_speed_kmh, config_.approach_max_speed_kmh);
  const bool slow_or_stopped_lead =
      valid_lead &&
      lead_speed_kmh <= std::max(0.0f, config_.stationary_lead_max_speed_kmh);

  switch (state_) {
    case AccStopState::Moving:
      if (low_speed_approach && slow_or_stopped_lead &&
          input.distance_m <= stop_trigger_distance_m) {
        TransitionTo(AccStopState::Stopping);
      }
      break;

    case AccStopState::Stopping:
      if (valid_lead) {
        target_lost_time_s_ = 0.0f;
      } else {
        target_lost_time_s_ += dt_s;
      }

      if (ego_speed_kmh <= std::max(0.0f, config_.hold_enter_speed_kmh)) {
        held_lead_id_ = valid_lead ? input.lead_id : -1;
        held_lead_distance_m_ = valid_lead ? input.distance_m : standstill_gap_m;
        TransitionTo(AccStopState::StoppedHold);
      }
      break;

    case AccStopState::StoppedHold:
      if (valid_lead) {
        target_lost_time_s_ = 0.0f;

        if (held_lead_id_ < 0) {
          held_lead_id_ = input.lead_id;
          held_lead_distance_m_ = input.distance_m;
        }

        if (input.lead_id != held_lead_id_) {
          held_lead_id_ = input.lead_id;
          held_lead_distance_m_ = input.distance_m;
          movement_confirm_time_s_ = 0.0f;
        }

        if (state_time_s_ >= std::max(0.0f, config_.minimum_hold_time_s) &&
            IsConfirmedLeadMovement(input)) {
          TransitionTo(AccStopState::ResumeConfirm);
        }
      } else {
        target_lost_time_s_ += dt_s;

        const bool minimum_hold_completed =
            state_time_s_ >= std::max(0.0f, config_.minimum_hold_time_s);
        const bool target_loss_confirmed =
            target_lost_time_s_ >=
            std::max(0.0f, config_.target_lost_resume_delay_s);

        if (minimum_hold_completed && target_loss_confirmed) {
          resume_without_lead_active_ = true;
          TransitionTo(AccStopState::Resuming);
        }
      }
      break;

    case AccStopState::ResumeConfirm:
      if (!valid_lead || input.lead_id != held_lead_id_) {
        TransitionTo(AccStopState::StoppedHold);
        break;
      }

      if (!IsConfirmedLeadMovement(input)) {
        TransitionTo(AccStopState::StoppedHold);
        break;
      }

      movement_confirm_time_s_ += dt_s;

      if (movement_confirm_time_s_ >=
          std::max(0.0f, config_.resume_confirm_time_s)) {
        TransitionTo(AccStopState::Resuming);
      }
      break;

    case AccStopState::Resuming:
      if (valid_lead) {
        target_lost_time_s_ = 0.0f;
      } else if (!resume_without_lead_active_) {
        target_lost_time_s_ += dt_s;
      }

      if (!resume_without_lead_active_ &&
          target_lost_time_s_ >=
          std::max(0.0f, config_.resume_target_lost_timeout_s)) {
        TransitionTo(ego_speed_kmh > config_.hold_enter_speed_kmh
                         ? AccStopState::Stopping
                         : AccStopState::StoppedHold);
        break;
      }

      if (valid_lead) {
        const float abort_gap_m =
            standstill_gap_m + std::max(0.0f, config_.close_gap_abort_margin_m);

        if (input.distance_m <= abort_gap_m ||
            lead_speed_kmh < config_.resume_lead_min_speed_kmh * 0.5f) {
          resume_without_lead_active_ = false;
          TransitionTo(AccStopState::Stopping);
          break;
        }
      }

      if (ego_speed_kmh >= config_.resume_exit_speed_kmh) {
        resume_without_lead_active_ = false;
        TransitionTo(AccStopState::Moving);
        break;
      }

      if (state_time_s_ >= std::max(0.1f, config_.resume_timeout_s)) {
        resume_without_lead_active_ = false;
        TransitionTo(ego_speed_kmh > config_.hold_enter_speed_kmh
                         ? AccStopState::Stopping
                         : AccStopState::StoppedHold);
        break;
      }
      break;
  }

  output.state = state_;

  switch (state_) {
    case AccStopState::Moving:
      output.accel_cmd_mps2 = input.base_accel_mps2;
      break;

    case AccStopState::Stopping: {
      const float remaining_distance_m =
          valid_lead ? std::max(0.20f, input.distance_m - standstill_gap_m)
                     : 0.20f;
      const float ego_speed_mps = std::max(0.0f, input.ego_speed_mps);
      const float required_decel_mps2 =
          ego_speed_mps * ego_speed_mps / (2.0f * remaining_distance_m);
      const float stop_decel_mps2 =
          Clamp(required_decel_mps2,
                std::max(0.05f, config_.stop_min_decel_mps2),
                std::max(config_.stop_min_decel_mps2,
                         config_.stop_max_decel_mps2));

      output.accel_cmd_mps2 =
          std::min(input.base_accel_mps2, -stop_decel_mps2);
      output.inhibit_throttle = true;
      break;
    }

    case AccStopState::StoppedHold:
    case AccStopState::ResumeConfirm:
      output.accel_cmd_mps2 = 0.0f;
      output.target_speed_kmh = 0.0f;
      output.force_hold_brake = true;
      output.hold_brake_0_10 = std::max(0.0f, config_.hold_brake_0_10);
      output.inhibit_throttle = true;
      break;

    case AccStopState::Resuming: {
      const float positive_base_accel = std::max(0.0f, input.base_accel_mps2);
      output.accel_cmd_mps2 =
          std::min(positive_base_accel,
                   std::max(0.0f, config_.resume_max_accel_mps2));

      const float valid_cruise_speed_kmh =
          std::max(0.0f, input.cruise_speed_kmh);
      const float valid_resume_cap_kmh =
          std::max(config_.resume_target_min_kmh, config_.resume_target_cap_kmh);

      if (resume_without_lead_active_ && !valid_lead) {
        output.target_speed_kmh =
            std::min(valid_cruise_speed_kmh, valid_resume_cap_kmh);
      } else {
        const float lead_based_target_kmh =
            lead_speed_kmh + std::max(0.0f, config_.resume_target_lead_margin_kmh);
        const float resume_target_kmh =
            std::max(config_.resume_target_min_kmh, lead_based_target_kmh);
        output.target_speed_kmh =
            std::min(valid_cruise_speed_kmh,
                     std::min(valid_resume_cap_kmh, resume_target_kmh));
      }
      output.inhibit_throttle = false;
      break;
    }
  }

  output.resume_without_lead_active = resume_without_lead_active_;
  output.state_time_s = state_time_s_;
  output.held_lead_id = held_lead_id_;
  output.held_lead_distance_m = held_lead_distance_m_;
  output.resume_confirm_time_s = movement_confirm_time_s_;

  return output;
}

float StopAndGoController::Clamp(const float value,
                                 const float lower,
                                 const float upper) noexcept {
  return std::max(lower, std::min(value, upper));
}

void StopAndGoController::TransitionTo(const AccStopState new_state) noexcept {
  if (new_state == state_) {
    return;
  }

  state_ = new_state;
  if (new_state != AccStopState::Resuming) {
    resume_without_lead_active_ = false;
  }
  state_time_s_ = 0.0f;
  movement_confirm_time_s_ = 0.0f;
  target_lost_time_s_ = 0.0f;
}

bool StopAndGoController::IsLeadValid(
    const StopAndGoInput& input) const noexcept {
  return input.has_lead &&
         input.lead_id >= 0 &&
         std::isfinite(input.distance_m) &&
         std::isfinite(input.lead_speed_mps) &&
         input.distance_m > 0.0f;
}

bool StopAndGoController::IsConfirmedLeadMovement(
    const StopAndGoInput& input) const noexcept {
  if (!IsLeadValid(input) || input.lead_id != held_lead_id_) {
    return false;
  }

  const float lead_speed_kmh =
      std::max(0.0f, input.lead_speed_mps) * kMpsToKmh;
  const float distance_increase_m = input.distance_m - held_lead_distance_m_;
  const float minimum_resume_gap_m =
      std::max(0.5f, input.standstill_gap_m) +
      std::max(0.0f, config_.resume_gap_margin_m);

  const bool lead_speed_confirmed =
      lead_speed_kmh >= std::max(0.0f, config_.resume_lead_min_speed_kmh);
  const bool distance_confirmed =
      distance_increase_m >= std::max(0.0f, config_.resume_distance_delta_m);

  return input.distance_m >= minimum_resume_gap_m &&
         (lead_speed_confirmed || distance_confirmed);
}

}  // namespace acc
