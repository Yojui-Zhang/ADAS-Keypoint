#include "runtime_control_commands.h"

#include <string>

namespace controller {
namespace {

std::string ToggleText(bool enabled) {
  return enabled ? "ON" : "OFF";
}

void FlipBool(bool* value) {
  if (value == nullptr) {
    return;
  }
  *value = (*value == false);
}

}  // namespace

bool HandleCommand(user_command_mode_t command,
                   RuntimeControlState* state,
                   std::string* out_message) {
  if (state == nullptr) {
    return false;
  }

  auto set_message = [&](const std::string& msg) {
    if (out_message == nullptr) {
      return;
    }
    *out_message = msg;
  };

  switch (command) {
    case CMD_1:
      if (state->canbus_compiled == false) {
        set_message("CANBus not compiled.");
        return false;
      }
      FlipBool(&state->can_tx_master_enable);
      set_message("CAN TX master -> " + ToggleText(state->can_tx_master_enable));
      return true;
    case CMD_2:
    case CMD_PLUS:
      if (state->canbus_compiled == false) {
        set_message("CANBus not compiled.");
        return false;
      }
      FlipBool(&state->can_throttle_enable);
      set_message("Throttle control -> " + ToggleText(state->can_throttle_enable));
      return true;
    case CMD_MINUS:
    case CMD_B:
      if (state->canbus_compiled == false) {
        set_message("CANBus not compiled.");
        return false;
      }
      FlipBool(&state->can_brake_enable);
      set_message("Brake control -> " + ToggleText(state->can_brake_enable));
      return true;
    case CMD_3:
      if (state->canbus_compiled == false) {
        set_message("CANBus not compiled.");
        return false;
      }
      FlipBool(&state->can_steering_enable);
      set_message("Steering control -> " + ToggleText(state->can_steering_enable));
      return true;
    case CMD_4:
      FlipBool(&state->draw_inference_overlay);
      set_message("Inference overlay -> " + ToggleText(state->draw_inference_overlay));
      return true;
    case CMD_5:
      FlipBool(&state->draw_acc_overlay);
      set_message("ACC overlay -> " + ToggleText(state->draw_acc_overlay));
      return true;
    case CMD_6:
      FlipBool(&state->draw_lka_overlay);
      set_message("LKA overlay -> " + ToggleText(state->draw_lka_overlay));
      return true;
    case CMD_7:
      FlipBool(&state->draw_behavior_overlay);
      set_message("Behavior overlay -> " + ToggleText(state->draw_behavior_overlay));
      return true;
    case CMD_8:
      FlipBool(&state->draw_collision_overlay);
      set_message("Collision overlay -> " + ToggleText(state->draw_collision_overlay));
      return true;
    case CMD_9:
      FlipBool(&state->draw_status_hud);
      set_message("HUD overlay -> " + ToggleText(state->draw_status_hud));
      return true;
    case CMD_G:
      FlipBool(&state->draw_ground_grid_overlay);
      set_message("Ground grid overlay -> " + ToggleText(state->draw_ground_grid_overlay));
      return true;
    case CMD_H:
      FlipBool(&state->draw_lane_detect_overlay);
      set_message("Lane detect overlay -> " + ToggleText(state->draw_lane_detect_overlay));
      return true;
    case CMD_D:
      FlipBool(&state->demo_presentation_mode);
      set_message("Demo presentation mode -> " + ToggleText(state->demo_presentation_mode));
      return true;
    case CMD_Q:
      FlipBool(&state->demo_lateral_control_enable);
      set_message("Demo lateral control -> " + ToggleText(state->demo_lateral_control_enable));
      return true;
    case CMD_W:
      FlipBool(&state->demo_longitudinal_control_enable);
      set_message("Demo longitudinal control -> " + ToggleText(state->demo_longitudinal_control_enable));
      return true;
    case CMD_E:
      FlipBool(&state->demo_supervisor_enable);
      set_message("Demo supervisor -> " + ToggleText(state->demo_supervisor_enable));
      return true;
    case CMD_R:
      FlipBool(&state->demo_lane_departure_warning_enable);
      set_message("Demo lane departure warning -> " +
                  ToggleText(state->demo_lane_departure_warning_enable));
      return true;
    case CMD_0: {
      const bool enable = (state->draw_inference_overlay &&
                           state->draw_acc_overlay &&
                           state->draw_lka_overlay &&
                           state->draw_behavior_overlay &&
                           state->draw_collision_overlay &&
                           state->draw_ground_grid_overlay &&
                           state->draw_lane_detect_overlay) == false;
      state->draw_inference_overlay = enable;
      state->draw_acc_overlay = enable;
      state->draw_lka_overlay = enable;
      state->draw_behavior_overlay = enable;
      state->draw_collision_overlay = enable;
      state->draw_ground_grid_overlay = enable;
      state->draw_lane_detect_overlay = enable;
      set_message(std::string("All overlays -> ") + ToggleText(enable));
      return true;
    }
    case CMD_RETURN:
      state->can_tx_master_enable = false;
      state->can_throttle_enable = false;
      state->can_brake_enable = false;
      state->can_steering_enable = false;
      set_message("Vehicle control outputs forced OFF.");
      return true;
    default:
      return false;
  }
}

}  // namespace controller
