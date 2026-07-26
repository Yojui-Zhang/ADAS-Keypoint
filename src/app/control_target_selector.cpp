#include "control_target_selector.h"

#include <algorithm>
#include <cmath>

#include "AccApi.h"

namespace adas_app {
namespace {

constexpr float kBrakeCommandThreshold = 0.05f;
constexpr float kValidSpeedThresholdKmh = 0.20f;
constexpr float kDecelerationOverrideMarginKmh = 2.0f;

constexpr float kMinimumLookaheadKmh = 3.0f;
constexpr float kMaximumLookaheadKmh = 8.0f;

constexpr float kLowSpeedBaseLookaheadKmh = 2.5f;
constexpr float kMaximumBaseLookaheadKmh = 6.0f;
constexpr float kSpeedLookaheadGain = 0.04f;

constexpr float kMaximumScheduledAccelMps2 = 2.0f;
constexpr float kAccelLookaheadGainKmhPerMps2 = 1.5f;
constexpr float kPositiveAccelThresholdMps2 = 0.05f;

float SanitizeSpeedKmh(const float speed_kmh) noexcept {
  if (!std::isfinite(speed_kmh)) {
    return 0.0f;
  }

  return std::max(0.0f, speed_kmh);
}

float SanitizeAccelerationMps2(const float acceleration_mps2) noexcept {
  if (!std::isfinite(acceleration_mps2)) {
    return 0.0f;
  }

  return acceleration_mps2;
}

float ComputeAccelerationLookaheadKmh(
    const float ego_speed_kmh,
    const float acceleration_command_mps2) noexcept {
  const float valid_ego_speed_kmh = SanitizeSpeedKmh(ego_speed_kmh);
  const float valid_acceleration_mps2 =
      std::clamp(SanitizeAccelerationMps2(acceleration_command_mps2),
                 0.0f,
                 kMaximumScheduledAccelMps2);

  const float speed_based_lookahead_kmh =
      std::clamp(kLowSpeedBaseLookaheadKmh +
                     kSpeedLookaheadGain * valid_ego_speed_kmh,
                 kLowSpeedBaseLookaheadKmh,
                 kMaximumBaseLookaheadKmh);

  const float acceleration_based_lookahead_kmh =
      kAccelLookaheadGainKmhPerMps2 * valid_acceleration_mps2;

  return std::clamp(speed_based_lookahead_kmh +
                        acceleration_based_lookahead_kmh,
                    kMinimumLookaheadKmh,
                    kMaximumLookaheadKmh);
}

float SelectRawPidTargetKmh(
    const stability::VehicleControlCommand& cmd,
    const float ego_speed_kmh) noexcept {
  const acc::AccCommand& acc_cmd = cmd.acc_cmd;

  const float valid_ego_speed_kmh = SanitizeSpeedKmh(ego_speed_kmh);
  const float supervisor_speed_kmh = SanitizeSpeedKmh(cmd.speed_kmh);

  // 曲率限制必須優先於駕駛巡航設定，避免後級重新繞過 Supervisor。
  if (cmd.stability_curve_is_bottleneck &&
      supervisor_speed_kmh > kValidSpeedThresholdKmh) {
    return supervisor_speed_kmh;
  }

  // 保留既有追車恢復目標來源，只在後續限制其瞬時前視範圍。
  if (acc_cmd.opening_recovery_active &&
      acc_cmd.opening_recovery_target_speed_kmh >
          kValidSpeedThresholdKmh) {
    return SanitizeSpeedKmh(
        acc_cmd.opening_recovery_target_speed_kmh);
  }

  // 明確減速情境維持原始前車速度目標，不套用正向速度前視窗。
  if (acc_cmd.has_lead &&
      acc_cmd.lead_following_active &&
      std::isfinite(acc_cmd.TargetSpeedKmh) &&
      acc_cmd.TargetSpeedKmh > kValidSpeedThresholdKmh &&
      acc_cmd.TargetSpeedKmh +
              kDecelerationOverrideMarginKmh <
          valid_ego_speed_kmh) {
    return SanitizeSpeedKmh(acc_cmd.TargetSpeedKmh);
  }

  if (acc_cmd.cruise_speed_kmh > kValidSpeedThresholdKmh) {
    return SanitizeSpeedKmh(acc_cmd.cruise_speed_kmh);
  }

  return supervisor_speed_kmh;
}

float GovernPositiveSpeedTargetKmh(
    const float raw_target_speed_kmh,
    const float supervisor_speed_kmh,
    const float ego_speed_kmh,
    const float acceleration_command_mps2) noexcept {
  const float valid_raw_target_kmh =
      SanitizeSpeedKmh(raw_target_speed_kmh);
  const float valid_supervisor_speed_kmh =
      SanitizeSpeedKmh(supervisor_speed_kmh);
  const float valid_ego_speed_kmh =
      SanitizeSpeedKmh(ego_speed_kmh);
  const float valid_acceleration_mps2 =
      SanitizeAccelerationMps2(acceleration_command_mps2);

  // 減速目標不得被正向前視限制抬高。
  if (valid_raw_target_kmh <= valid_ego_speed_kmh) {
    return valid_raw_target_kmh;
  }

  // Supervisor 沒有要求正向加速時，不允許巡航設定自行產生推力。
  if (valid_acceleration_mps2 <= kPositiveAccelThresholdMps2) {
    const float hold_target_kmh =
        std::max(valid_ego_speed_kmh,
                 valid_supervisor_speed_kmh);

    return std::min(valid_raw_target_kmh, hold_target_kmh);
  }

  const float lookahead_kmh =
      ComputeAccelerationLookaheadKmh(valid_ego_speed_kmh,
                                      valid_acceleration_mps2);

  const float maximum_tracking_target_kmh =
      valid_ego_speed_kmh + lookahead_kmh;

  return std::min(valid_raw_target_kmh,
                  maximum_tracking_target_kmh);
}

}  // namespace

float SelectActuatorSpeedTargetKmh(
    const controller::RuntimeControlState& control_state,
    const stability::VehicleControlCommand& cmd,
    const float ego_speed_kmh) {
  if (!controller::DemoLongitudinalControlEnabled(control_state)) {
    return 0.0f;
  }

  if (control_state.longitudinal_controller !=
      controller::LongitudinalControllerKind::Pid) {
    return SanitizeSpeedKmh(cmd.speed_kmh);
  }

  const acc::AccCommand& acc_cmd = cmd.acc_cmd;

  if (cmd.brake_0_10 > kBrakeCommandThreshold ||
      acc_cmd.longitudinal_phase ==
          acc::AccLongitudinalPhase::Braking) {
    return 0.0f;
  }

  if (acc_cmd.longitudinal_phase ==
      acc::AccLongitudinalPhase::Idle) {
    return 0.0f;
  }

  if (acc_cmd.longitudinal_phase ==
      acc::AccLongitudinalPhase::Coasting) {
    return SanitizeSpeedKmh(cmd.speed_kmh);
  }

  // Stop-and-Go 已有獨立的低速加速度與目標上限，不應再改變其行為。
  if (acc_cmd.stop_state == acc::AccStopState::Resuming) {
    return SanitizeSpeedKmh(cmd.speed_kmh);
  }

  const float raw_target_speed_kmh =
      SelectRawPidTargetKmh(cmd, ego_speed_kmh);

  return GovernPositiveSpeedTargetKmh(
      raw_target_speed_kmh,
      cmd.speed_kmh,
      ego_speed_kmh,
      cmd.stability_a_long_cmd_mps2);
}

}  // namespace adas_app
