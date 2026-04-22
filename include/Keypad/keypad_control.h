#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "user_command.h"

struct AppRuntimeConfig;

namespace keypad {

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
};

RuntimeControlState MakeInitialRuntimeControlState(const AppRuntimeConfig& cfg,
                                                   bool canbus_compiled);

bool HandleCommand(user_command_mode_t command,
                   RuntimeControlState* state,
                   std::string* out_message = nullptr);

void SyncCanRuntimeState(const RuntimeControlState& state);
void ShutdownRuntimeControl(RuntimeControlState* state);
void DrawRuntimeStatusOverlay(cv::Mat& frame,
                              const RuntimeControlState& state,
                              bool evdev_ready);

bool LongitudinalControlActive(const RuntimeControlState& state);
bool SteeringControlActive(const RuntimeControlState& state);
bool ThrottleControlActive(const RuntimeControlState& state);
bool BrakeControlActive(const RuntimeControlState& state);

}  // namespace keypad
