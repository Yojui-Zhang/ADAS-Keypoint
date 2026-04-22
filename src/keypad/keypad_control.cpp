#include "keypad_control.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "config.h"
#include "system_config.h"

#ifdef CANBUS__
#include "canbus_recv.h"
#endif

#ifdef USE_ITRI_CAN
#include "pid_controller.h"
#endif

#ifdef CANBUS__
extern CAR CAN;
extern float target_speed;
extern volatile double deceleration;
#endif

namespace keypad {

namespace {

std::string ToggleText(bool enabled) {
  return enabled ? "ON" : "OFF";
}

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

#ifdef CANBUS__
struct SpeedPidGains {
  float kp = 5.4f;
  float ki = 30.0f;
  float kd = 2.5f;
};

struct LongitudinalTxState {
  std::atomic<bool> throttle_running{false};
  std::thread throttle_worker;
  bool brake_sender_started = false;
  LongitudinalControllerKind controller_kind = LongitudinalControllerKind::Keypad;
};

LongitudinalTxState& GetLongitudinalTxState() {
  static LongitudinalTxState state;
  return state;
}

double ClampValue(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

SpeedPidGains SelectSpeedPidGains(double speed_kmh) {
  SpeedPidGains gains;
  if (speed_kmh <= 20.0) {
    gains.kp = 1.25f;
    gains.ki = 3.5f;
    gains.kd = 1.88f;
  } else if (speed_kmh <= 30.0) {
    gains.kp = 1.8f;
    gains.ki = 9.0f;
    gains.kd = 1.9f;
  } else if (speed_kmh <= 40.0) {
    gains.kp = 2.25f;
    gains.ki = 14.5f;
    gains.kd = 1.95f;
  } else if (speed_kmh <= 50.0) {
    gains.kp = 2.8f;
    gains.ki = 17.0f;
    gains.kd = 2.1f;
  } else if (speed_kmh <= 60.0) {
    gains.kp = 3.4f;
    gains.ki = 22.0f;
    gains.kd = 2.30f;
  } else if (speed_kmh <= 70.0) {
    gains.kp = 4.6f;
    gains.ki = 27.5f;
    gains.kd = 2.5f;
  } else {
    gains.kp = 5.4f;
    gains.ki = 32.0f;
    gains.kd = 2.5f;
  }
  return gains;
}

double SelectSpeedPidPedalUpperLimit(double speed_kmh) {
  if (speed_kmh <= 20.0) return 1.60;
  if (speed_kmh <= 30.0) return 2.05;
  if (speed_kmh <= 40.0) return 2.40;
  return 2.80;
}

float LimitSpeedPidTarget(float desired_speed_kmh, float current_speed_kmh) {
  const float speed_error_kmh = desired_speed_kmh - current_speed_kmh;
  if (speed_error_kmh <= 0.0f) {
    return desired_speed_kmh;
  }

  float max_visible_error_kmh = 8.0f;
  if (current_speed_kmh > 60.0f) {
    max_visible_error_kmh = 18.0f;
  } else if (current_speed_kmh > 40.0f) {
    max_visible_error_kmh = 14.0f;
  } else if (current_speed_kmh > 20.0f) {
    max_visible_error_kmh = 10.0f;
  }

  return current_speed_kmh + std::min(speed_error_kmh, max_visible_error_kmh);
}

void StopThrottleThread() {
  LongitudinalTxState& state = GetLongitudinalTxState();
  state.throttle_running.store(false, std::memory_order_release);
  if (state.throttle_worker.joinable()) {
    state.throttle_worker.join();
  }
}

void StartThrottleThread(LongitudinalControllerKind controller_kind) {
  LongitudinalTxState& state = GetLongitudinalTxState();
  if (state.throttle_running.load(std::memory_order_acquire)) {
    return;
  }

  state.controller_kind = controller_kind;
  state.throttle_running.store(true, std::memory_order_release);
  state.throttle_worker = std::thread([controller_kind]() {
    if (controller_kind == LongitudinalControllerKind::Pid) {
// #ifdef USE_ITRI_CAN
      PID_incremental throttle;


      while (GetLongitudinalTxState().throttle_running.load(std::memory_order_acquire)) {
        const float desired_speed_kmh = std::max(0.0f, ::target_speed);
        const float current_speed_kmh = static_cast<float>(std::max(0.0, ::CAN.speed));
        const float pid_target_speed_kmh =
            LimitSpeedPidTarget(desired_speed_kmh, current_speed_kmh);
        const bool braking_now = ::deceleration > 0.05;
        const SpeedPidGains gains = SelectSpeedPidGains(current_speed_kmh);
        const double pedal_upper_limit = SelectSpeedPidPedalUpperLimit(current_speed_kmh);

        double pedal_cmd = 0.75;
        if (!braking_now && desired_speed_kmh > 0.2f) {
          pedal_cmd = throttle.pid_control_ACC(pid_target_speed_kmh,
                                               current_speed_kmh,
                                               gains.kp,
                                               gains.ki,
                                               gains.kd);
          pedal_cmd = ClampValue(pedal_cmd, 0.75, pedal_upper_limit);
        } else {
          throttle.e_pre_1 = 0.0f;
          throttle.e_pre_2 = 0.0f;
        }

        canbus_ctrl_pedal(pedal_cmd);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      return;
// #endif
    }

    while (GetLongitudinalTxState().throttle_running.load(std::memory_order_acquire)) {
      const double desired_speed_kmh = std::max(0.0f, ::target_speed);
      const double current_speed_kmh = std::max(0.0, ::CAN.speed);
      const bool braking_now = ::deceleration > 0.05;

      double pedal_cmd = 0.75;
      if (desired_speed_kmh > 0.2 && braking_now == false) {
        const double speed_error = desired_speed_kmh - current_speed_kmh;
        if (speed_error > 0.2) {
          pedal_cmd = 0.75 + ClampValue(speed_error * 2.56, 0.0, 2.05);
        }
      }

      canbus_ctrl_pedal(pedal_cmd);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
}

void ApplyThrottleRuntime(bool active, LongitudinalControllerKind controller_kind) {
  if (active) {
    StartThrottleThread(controller_kind);
    return;
  }

  ::target_speed = 0.0f;
  canbus_ctrl_pedal(0.75);
  StopThrottleThread();
}

void ApplyBrakeRuntime(bool active) {
  LongitudinalTxState& state = GetLongitudinalTxState();
  if (active) {
    if (state.brake_sender_started == false) {
      canbus_ctrl_dec(1);
      state.brake_sender_started = true;
    }
    return;
  }

  ::deceleration = 0.0;
  if (state.brake_sender_started) {
    canbus_ctrl_dec(0);
    state.brake_sender_started = false;
  }
}
#endif

void FlipBool(bool* value) {
  if (value == nullptr) {
    return;
  }
  *value = (*value == false);
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

void SyncCanRuntimeState(const RuntimeControlState& state) {
#ifdef CANBUS__
  static bool last_throttle = false;
  static bool last_brake = false;
  static bool last_steering = false;

  const bool throttle_active = ThrottleControlActive(state);
  const bool brake_active = BrakeControlActive(state);
  const bool steering_active = SteeringControlActive(state);

  if (throttle_active == last_throttle) {
  } else {
    ApplyThrottleRuntime(throttle_active, state.longitudinal_controller);
    last_throttle = throttle_active;
  }

  if (brake_active == last_brake) {
  } else {
    ApplyBrakeRuntime(brake_active);
    last_brake = brake_active;
  }

  if (steering_active == last_steering) {
  } else {
    canbus_set_steering_tx_enabled(steering_active ? 1 : 0);
    last_steering = steering_active;
  }
#else
  (void)state;
#endif
}

void ShutdownRuntimeControl(RuntimeControlState* state) {
  if (state == nullptr) {
    return;
  }

  state->can_tx_master_enable = false;
  state->can_throttle_enable = false;
  state->can_brake_enable = false;
  state->can_steering_enable = false;
  SyncCanRuntimeState(*state);
}

void DrawRuntimeStatusOverlay(cv::Mat& frame,
                              const RuntimeControlState& state,
                              bool evdev_ready) {
  if (frame.empty() || state.draw_status_hud == false) {
    return;
  }

  std::vector<std::string> lines;
  lines.emplace_back("Hotkeys 1:TX 2/+:Throttle -/B:Brake 3:Steer 4:Infer 5:ACC 6:LKA 7:Behavior 8:Collision 9:HUD G:Grid H:LaneDet 0:All Backspace:SafeOff\n\n");
  lines.emplace_back("CAN compile:" + ToggleText(state.canbus_compiled));
  lines.emplace_back("keypad:" + ToggleText(evdev_ready));
  lines.emplace_back("TX master:" + ToggleText(state.can_tx_master_enable));
  lines.emplace_back("longitudinal:" + ToggleText(LongitudinalControlActive(state)));
  lines.emplace_back("longitudinal ctl:" + state.longitudinal_controller_name);
  lines.emplace_back("steer:" + ToggleText(SteeringControlActive(state)));
  lines.emplace_back("Throttle:" + ToggleText(ThrottleControlActive(state)));
  lines.emplace_back("Brake:" + ToggleText(BrakeControlActive(state)));
  lines.emplace_back("Steering:" + ToggleText(SteeringControlActive(state)));
  lines.emplace_back("Draw infer:" + ToggleText(state.draw_inference_overlay));
  lines.emplace_back("ACC:" + ToggleText(state.draw_acc_overlay));
  lines.emplace_back("LKA:" + ToggleText(state.draw_lka_overlay));
  lines.emplace_back("Behavior:" + ToggleText(state.draw_behavior_overlay));
  lines.emplace_back("Collision:" + ToggleText(state.draw_collision_overlay));
  lines.emplace_back("Grid:" + ToggleText(state.draw_ground_grid_overlay));
  lines.emplace_back("LaneDet:" + ToggleText(state.draw_lane_detect_overlay));

  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const double scale = 0.55;
  const int thickness = 1;
  const int left = 20;
  const int top = 380;

  // int panel_width = 0;
  // int panel_height = padding;
  // for (const auto& line : lines) {
  //   const cv::Size sz = cv::getTextSize(line, font, scale, thickness, &baseline);
  //   panel_width = std::max(panel_width, sz.width);
  //   panel_height += sz.height + 8;
  // }

  // panel_height += padding;
  // const cv::Rect panel(left - padding,
  //                      top - padding,
  //                      panel_width + padding * 2,
  //                      panel_height);
  // cv::rectangle(frame, panel, cv::Scalar(20, 20, 20), cv::FILLED);
  // cv::rectangle(frame, panel, cv::Scalar(90, 90, 90), 1);

  int y = top;
  for (const auto& line : lines) {
    cv::putText(frame,
                line,
                cv::Point(left, y),
                font,
                scale,
                BLACK,
                thickness+1,
                cv::LINE_AA);
    cv::putText(frame,
                line,
                cv::Point(left, y),
                font,
                scale,
                WHITE,
                thickness,
                cv::LINE_AA);
    y += 22;
  }
}

}  // namespace keypad
