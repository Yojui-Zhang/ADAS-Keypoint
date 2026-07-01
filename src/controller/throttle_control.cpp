#include "throttle_control.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include "speed_pid_controller.h"

#ifdef CANBUS__
#include "canbus_recv.h"

extern CAR CAN;
extern float target_speed;
extern volatile double deceleration;
#endif

namespace controller {
namespace {

#ifdef CANBUS__
struct ThrottleTxState {
  std::atomic<bool> running{false};
  std::thread worker;
  LongitudinalControllerKind controller_kind = LongitudinalControllerKind::Keypad;
};

ThrottleTxState& GetThrottleTxState() {
  static ThrottleTxState state;
  return state;
}

void StopThrottleThread() {
  ThrottleTxState& state = GetThrottleTxState();
  state.running.store(false, std::memory_order_release);
  if (state.worker.joinable()) {
    state.worker.join();
  }
}

void StartThrottleThread(LongitudinalControllerKind controller_kind) {
  ThrottleTxState& state = GetThrottleTxState();
  if (state.running.load(std::memory_order_acquire)) {
    return;
  }

  state.controller_kind = controller_kind;
  state.running.store(true, std::memory_order_release);
  state.worker = std::thread([controller_kind]() {
    if (controller_kind == LongitudinalControllerKind::Pid) {
      IncrementalSpeedPid throttle;

      while (GetThrottleTxState().running.load(std::memory_order_acquire)) {
        const float desired_speed_kmh = std::max(0.0f, ::target_speed);
        const float current_speed_kmh = static_cast<float>(std::max(0.0, ::CAN.speed));
        const float pid_target_speed_kmh =
            LimitSpeedPidTarget(desired_speed_kmh, current_speed_kmh);
        const bool braking_now = ::deceleration > 0.05;
        const SpeedPidGains gains = SelectSpeedPidGains(current_speed_kmh);
        const double pedal_upper_limit = SelectSpeedPidPedalUpperLimit(current_speed_kmh);

        double pedal_cmd = 0.75;
        if (!braking_now && desired_speed_kmh > 0.2f) {
          pedal_cmd = throttle.Compute(pid_target_speed_kmh, current_speed_kmh, gains);
          pedal_cmd = ClampControllerValue(pedal_cmd, 0.75, pedal_upper_limit);
        } else {
          throttle.Reset();
        }

        canbus_ctrl_pedal(pedal_cmd);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      return;
    }

    while (GetThrottleTxState().running.load(std::memory_order_acquire)) {
      const double desired_speed_kmh = std::max(0.0f, ::target_speed);
      const double current_speed_kmh = std::max(0.0, ::CAN.speed);
      const bool braking_now = ::deceleration > 0.05;

      double pedal_cmd = 0.75;
      if (desired_speed_kmh > 0.2 && braking_now == false) {
        const double speed_error = desired_speed_kmh - current_speed_kmh;
        if (speed_error > 0.2) {
          pedal_cmd = 0.75 + ClampControllerValue(speed_error * 2.56, 0.0, 2.05);
        }
      }

      canbus_ctrl_pedal(pedal_cmd);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
}
#endif

}  // namespace

void ApplyThrottleRuntime(bool active, LongitudinalControllerKind controller_kind) {
#ifdef CANBUS__
  if (active) {
    StartThrottleThread(controller_kind);
    return;
  }

  ::target_speed = 0.0f;
  canbus_ctrl_pedal(0.75);
  StopThrottleThread();
#else
  (void)active;
  (void)controller_kind;
#endif
}

void StopThrottleRuntime() {
#ifdef CANBUS__
  StopThrottleThread();
#endif
}

}  // namespace controller
