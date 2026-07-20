#pragma once

#include <cstdint>

namespace acc {

enum class AccStopState : std::uint8_t {
  Moving = 0,
  Stopping = 1,
  StoppedHold = 2,
  ResumeConfirm = 3,
  Resuming = 4,
};

inline const char* AccStopStateName(const AccStopState state) noexcept {
  switch (state) {
    case AccStopState::Moving:
      return "moving";
    case AccStopState::Stopping:
      return "stopping";
    case AccStopState::StoppedHold:
      return "stopped_hold";
    case AccStopState::ResumeConfirm:
      return "resume_confirm";
    case AccStopState::Resuming:
      return "resuming";
  }

  return "unknown";
}

inline int AccStopStateCode(const AccStopState state) noexcept {
  return static_cast<int>(state);
}

struct StopAndGoConfig {
  bool enabled = true;

  float approach_max_speed_kmh = 12.0f;
  float stationary_lead_max_speed_kmh = 2.0f;

  float stop_trigger_margin_m = 3.0f;
  float stop_trigger_time_s = 0.8f;

  float stop_min_decel_mps2 = 0.35f;
  float stop_max_decel_mps2 = 1.20f;

  float hold_enter_speed_kmh = 0.50f;
  float hold_brake_0_10 = 0.35f;
  float minimum_hold_time_s = 1.0f;
  float target_lost_resume_delay_s = 0.8f;

  float resume_lead_min_speed_kmh = 2.0f;
  float resume_distance_delta_m = 1.0f;
  float resume_confirm_time_s = 0.60f;
  float resume_gap_margin_m = 0.80f;

  float resume_max_accel_mps2 = 0.60f;
  float resume_target_min_kmh = 5.0f;
  float resume_target_lead_margin_kmh = 3.0f;
  float resume_target_cap_kmh = 12.0f;

  float resume_exit_speed_kmh = 8.0f;
  float resume_timeout_s = 4.0f;

  float close_gap_abort_margin_m = 0.30f;
  float resume_target_lost_timeout_s = 0.30f;
};

struct StopAndGoInput {
  bool has_lead = false;
  int lead_id = -1;

  float ego_speed_mps = 0.0f;
  float lead_speed_mps = 0.0f;
  float distance_m = 0.0f;

  float standstill_gap_m = 4.0f;
  float cruise_speed_kmh = 0.0f;

  float base_accel_mps2 = 0.0f;
  float dt_s = 0.05f;
};

struct StopAndGoOutput {
  AccStopState state = AccStopState::Moving;

  float accel_cmd_mps2 = 0.0f;
  float target_speed_kmh = 0.0f;

  bool force_hold_brake = false;
  float hold_brake_0_10 = 0.0f;

  bool inhibit_throttle = false;
  bool resume_without_lead_active = false;

  float state_time_s = 0.0f;
  int held_lead_id = -1;
  float held_lead_distance_m = 0.0f;
  float resume_confirm_time_s = 0.0f;
};

class StopAndGoController {
 public:
  explicit StopAndGoController(StopAndGoConfig config = {});

  void SetConfig(const StopAndGoConfig& config);

  [[nodiscard]] const StopAndGoConfig& GetConfig() const noexcept;

  StopAndGoOutput Update(const StopAndGoInput& input);

  void Reset() noexcept;

  void RequestManualResume() noexcept;

  [[nodiscard]] AccStopState State() const noexcept;

 private:
  static float Clamp(float value, float lower, float upper) noexcept;

  void TransitionTo(AccStopState new_state) noexcept;

  bool IsLeadValid(const StopAndGoInput& input) const noexcept;

  bool IsConfirmedLeadMovement(const StopAndGoInput& input) const noexcept;

  StopAndGoConfig config_{};

  AccStopState state_ = AccStopState::Moving;

  float state_time_s_ = 0.0f;
  float movement_confirm_time_s_ = 0.0f;
  float target_lost_time_s_ = 0.0f;

  int held_lead_id_ = -1;
  float held_lead_distance_m_ = 0.0f;

  bool manual_resume_requested_ = false;
  bool resume_without_lead_active_ = false;
};

}  // namespace acc
