#pragma once

#include <cstdint>
#include <string>

struct AppRuntimeConfig;

namespace controller {

enum class LongitudinalControllerKind {
  Keypad,
  Pid,
};

struct RuntimeControlState {
  bool canbus_compiled = false;

  bool can_tx_master_enable = false;
  bool can_throttle_enable = false;
  bool can_brake_enable = false;
  bool can_steering_enable = false;
  LongitudinalControllerKind longitudinal_controller = LongitudinalControllerKind::Keypad;
  std::string longitudinal_controller_name = "keypad";

  bool draw_inference_overlay = true;
  bool draw_acc_overlay = true;
  bool draw_lka_overlay = true;
  bool draw_behavior_overlay = true;
  bool draw_collision_overlay = true;
  bool draw_ground_grid_overlay = false;
  bool draw_lane_detect_overlay = false;
  bool draw_status_hud = true;

  bool demo_presentation_mode = false;
  bool demo_lateral_control_enable = true;
  bool demo_longitudinal_control_enable = true;
  bool demo_supervisor_enable = true;
  bool demo_lane_departure_warning_enable = true;

  bool acc_resume_request_pending = false;
  std::uint64_t acc_resume_request_sequence = 0;

  int traffic_light_override_class_id = -1;
  int speed_sign_override_id = -1;
  std::uint64_t speed_sign_override_sequence = 0;
};

RuntimeControlState MakeInitialRuntimeControlState(const AppRuntimeConfig& cfg,
                                                   bool canbus_compiled);

bool LongitudinalControlActive(const RuntimeControlState& state);
bool SteeringControlActive(const RuntimeControlState& state);
bool ThrottleControlActive(const RuntimeControlState& state);
bool BrakeControlActive(const RuntimeControlState& state);

bool DemoPresentationActive(const RuntimeControlState& state);
bool DemoLateralControlEnabled(const RuntimeControlState& state);
bool DemoLongitudinalControlEnabled(const RuntimeControlState& state);
bool DemoSupervisorEnabled(const RuntimeControlState& state);
bool DemoLaneDepartureWarningEnabled(const RuntimeControlState& state);

}  // namespace controller
