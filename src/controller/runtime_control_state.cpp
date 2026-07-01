#include "runtime_control_state.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "system_config.h"

namespace controller {
namespace {

std::string ToLowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

const char* LongitudinalControllerName(LongitudinalControllerKind kind) {
  switch (kind) {
    case LongitudinalControllerKind::Pid:
      return "pid";
    default:
      return "keypad";
  }
}

LongitudinalControllerKind ParseLongitudinalControllerKind(const std::string& value) {
  const std::string mode = ToLowerCopy(value);
  if (mode == "pid" || mode == "itri_pid" || mode == "pid_controller") {
    return LongitudinalControllerKind::Pid;
  }
  return LongitudinalControllerKind::Keypad;
}

}  // namespace

RuntimeControlState MakeInitialRuntimeControlState(const AppRuntimeConfig& cfg,
                                                   bool canbus_compiled) {
  RuntimeControlState state;
  state.canbus_compiled = canbus_compiled;

  state.can_tx_master_enable = canbus_compiled && cfg.can_tx_master_enable;
  const bool legacy_longitudinal_enable = canbus_compiled && cfg.can_longitudinal_enable;
  state.can_throttle_enable =
      canbus_compiled && (cfg.can_throttle_enable || legacy_longitudinal_enable);
  state.can_brake_enable =
      canbus_compiled && (cfg.can_brake_enable || legacy_longitudinal_enable);
  state.can_steering_enable = canbus_compiled && cfg.can_steering_enable;
  state.longitudinal_controller = ParseLongitudinalControllerKind(cfg.longitudinal_controller);
  state.longitudinal_controller_name = LongitudinalControllerName(state.longitudinal_controller);

  state.draw_inference_overlay = cfg.draw_inference_overlay;
  state.draw_acc_overlay = cfg.draw_acc_overlay;
  state.draw_lka_overlay = cfg.draw_lka_overlay;
  state.draw_behavior_overlay = cfg.draw_behavior_overlay;
  state.draw_collision_overlay = cfg.draw_collision_overlay;
  state.draw_ground_grid_overlay = cfg.draw_ground_grid_overlay;
  state.draw_lane_detect_overlay = cfg.draw_lane_detect_overlay;
  state.draw_status_hud = cfg.draw_status_hud;
  return state;
}

bool ThrottleControlActive(const RuntimeControlState& state) {
  return state.canbus_compiled &&
         state.can_tx_master_enable &&
         state.can_throttle_enable;
}

bool BrakeControlActive(const RuntimeControlState& state) {
  return state.canbus_compiled &&
         state.can_tx_master_enable &&
         state.can_brake_enable;
}

bool LongitudinalControlActive(const RuntimeControlState& state) {
  return ThrottleControlActive(state) || BrakeControlActive(state);
}

bool SteeringControlActive(const RuntimeControlState& state) {
  return state.canbus_compiled &&
         state.can_tx_master_enable &&
         state.can_steering_enable;
}

bool DemoPresentationActive(const RuntimeControlState& state) {
  return state.demo_presentation_mode;
}

bool DemoLateralControlEnabled(const RuntimeControlState& state) {
  return state.demo_presentation_mode == false || state.demo_lateral_control_enable;
}

bool DemoLongitudinalControlEnabled(const RuntimeControlState& state) {
  return state.demo_presentation_mode == false || state.demo_longitudinal_control_enable;
}

bool DemoSupervisorEnabled(const RuntimeControlState& state) {
  return state.demo_presentation_mode == false || state.demo_supervisor_enable;
}

bool DemoLaneDepartureWarningEnabled(const RuntimeControlState& state) {
  return state.demo_presentation_mode == false || state.demo_lane_departure_warning_enable;
}

}  // namespace controller
