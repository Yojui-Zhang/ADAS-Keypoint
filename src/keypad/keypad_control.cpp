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
struct LongitudinalTxState {
  std::atomic<bool> running{false};
  std::thread worker;
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

void StopLongitudinalThread() {
  LongitudinalTxState& state = GetLongitudinalTxState();
  state.running.store(false, std::memory_order_release);
  if (state.worker.joinable()) {
    state.worker.join();
  }
}

void StartLongitudinalThread(LongitudinalControllerKind controller_kind) {
  LongitudinalTxState& state = GetLongitudinalTxState();
  if (state.running.load(std::memory_order_acquire)) {
    return;
  }

  state.controller_kind = controller_kind;
  state.running.store(true, std::memory_order_release);
  state.worker = std::thread([controller_kind]() {
    if (controller_kind == LongitudinalControllerKind::Pid) {
#ifdef USE_ITRI_CAN
      PID_incremental throttle(0.031666667f, 0.5f, 0.8f);
      constexpr float kPidKp = 0.031666667f;
      constexpr float kPidKi = 0.5f;
      constexpr float kPidKd = 0.28f;

      while (GetLongitudinalTxState().running.load(std::memory_order_acquire)) {
        const float desired_speed_kmh = std::max(0.0f, ::target_speed);
        const float current_speed_kmh = static_cast<float>(std::max(0.0, ::CAN.speed));
        const bool braking_now = ::deceleration > 0.05;

        double pedal_cmd = 0.75;
        if (!braking_now && desired_speed_kmh > 0.2f) {
          pedal_cmd = throttle.pid_control_ACC(desired_speed_kmh,
                                               current_speed_kmh,
                                               kPidKp,
                                               kPidKi,
                                               kPidKd);
          pedal_cmd = ClampValue(pedal_cmd, 0.75, 2.8);
        }

        canbus_ctrl_pedal(pedal_cmd);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      return;
#endif
    }

    while (GetLongitudinalTxState().running.load(std::memory_order_acquire)) {
      const double desired_speed_kmh = std::max(0.0f, ::target_speed);
      const double current_speed_kmh = std::max(0.0, ::CAN.speed);
      const bool braking_now = ::deceleration > 0.05;

      double pedal_cmd = 0.75;
      if (desired_speed_kmh > 0.2 && braking_now == false) {
        const double speed_error = desired_speed_kmh - current_speed_kmh;
        if (speed_error > 0.2) {
          pedal_cmd = 0.75 + ClampValue(speed_error * 0.26, 0.0, 2.05);
        }
      }

      canbus_ctrl_pedal(pedal_cmd);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
}

void ApplyLongitudinalRuntime(bool active, LongitudinalControllerKind controller_kind) {
  LongitudinalTxState& state = GetLongitudinalTxState();
  if (active) {
    if (state.brake_sender_started == false) {
      canbus_ctrl_dec(1);
      state.brake_sender_started = true;
    }
    StartLongitudinalThread(controller_kind);
    return;
  }

  ::deceleration = 0.0;
  ::target_speed = 0.0f;
  canbus_ctrl_pedal(0.75);
  StopLongitudinalThread();

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
  state.can_longitudinal_enable = canbus_compiled && cfg.can_longitudinal_enable;
  state.can_steering_enable = canbus_compiled && cfg.can_steering_enable;
  state.longitudinal_controller = ParseLongitudinalControllerKind(cfg.longitudinal_controller);
  state.longitudinal_controller_name = LongitudinalControllerName(state.longitudinal_controller);

  state.draw_inference_overlay = cfg.draw_inference_overlay;
  state.draw_acc_overlay = cfg.draw_acc_overlay;
  state.draw_lka_overlay = cfg.draw_lka_overlay;
  state.draw_behavior_overlay = cfg.draw_behavior_overlay;
  state.draw_collision_overlay = cfg.draw_collision_overlay;
  state.draw_status_hud = cfg.draw_status_hud;
  return state;
}

bool LongitudinalControlActive(const RuntimeControlState& state) {
  return state.canbus_compiled &&
         state.can_tx_master_enable &&
         state.can_longitudinal_enable;
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
      if (state->canbus_compiled == false) {
        set_message("CANBus not compiled.");
        return false;
      }
      FlipBool(&state->can_longitudinal_enable);
      set_message("Longitudinal control -> " + ToggleText(state->can_longitudinal_enable));
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
    case CMD_0: {
      const bool enable = (state->draw_inference_overlay &&
                           state->draw_acc_overlay &&
                           state->draw_lka_overlay &&
                           state->draw_behavior_overlay &&
                           state->draw_collision_overlay) == false;
      state->draw_inference_overlay = enable;
      state->draw_acc_overlay = enable;
      state->draw_lka_overlay = enable;
      state->draw_behavior_overlay = enable;
      state->draw_collision_overlay = enable;
      set_message(std::string("All overlays -> ") + ToggleText(enable));
      return true;
    }
    case CMD_RETURN:
      state->can_tx_master_enable = false;
      state->can_longitudinal_enable = false;
      state->can_steering_enable = false;
      set_message("Vehicle control outputs forced OFF.");
      return true;
    default:
      return false;
  }
}

void SyncCanRuntimeState(const RuntimeControlState& state) {
#ifdef CANBUS__
  static bool last_longitudinal = false;
  static bool last_steering = false;

  const bool longitudinal_active = LongitudinalControlActive(state);
  const bool steering_active = SteeringControlActive(state);

  if (longitudinal_active == last_longitudinal) {
  } else {
    ApplyLongitudinalRuntime(longitudinal_active, state.longitudinal_controller);
    last_longitudinal = longitudinal_active;
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
  state->can_longitudinal_enable = false;
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
  lines.emplace_back("Hotkeys 1:TX 2:Speed/Brake 3:Steer 4:Infer 5:ACC 6:LKA 7:Behavior 8:Collision 9:HUD 0:All Backspace:SafeOff\n\n");
  lines.emplace_back("CAN compile:" + ToggleText(state.canbus_compiled));
  lines.emplace_back("keypad:" + ToggleText(evdev_ready));
  lines.emplace_back("TX master:" + ToggleText(state.can_tx_master_enable));
  lines.emplace_back("speed/brake:" + ToggleText(LongitudinalControlActive(state)));
  lines.emplace_back("longitudinal ctl:" + state.longitudinal_controller_name);
  lines.emplace_back("steer:" + ToggleText(SteeringControlActive(state)));
  lines.emplace_back("Throttle:" + ToggleText(LongitudinalControlActive(state)));
  lines.emplace_back("Brake:" + ToggleText(LongitudinalControlActive(state)));
  lines.emplace_back("Steering:" + ToggleText(SteeringControlActive(state)));
  lines.emplace_back("Draw infer:" + ToggleText(state.draw_inference_overlay));
  lines.emplace_back("ACC:" + ToggleText(state.draw_acc_overlay));
  lines.emplace_back("LKA:" + ToggleText(state.draw_lka_overlay));
  lines.emplace_back("Behavior:" + ToggleText(state.draw_behavior_overlay));
  lines.emplace_back("Collision:" + ToggleText(state.draw_collision_overlay));

  int baseline = 0;
  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const double scale = 0.55;
  const int thickness = 1;
  const int left = 20;
  const int top = 380;
  const int padding = 10;

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
