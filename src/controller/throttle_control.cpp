#include "throttle_control.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

#include "config.h"
#include "longitudinal_vehicle_state.h"
#include "speed_pid_controller.h"
#include "time_sync.h"

#ifdef CANBUS__
#include "canbus_recv.h"

extern float target_speed;
#endif

namespace controller {
namespace {

#ifdef CANBUS__
struct ThrottleTxState {
  std::atomic<bool> running{false};
  std::atomic<float> target_speed_kmh{0.0f};
  std::atomic<ThrottleControlMode> mode{ThrottleControlMode::Disabled};
  std::atomic<ThrottleControlMode> telemetry_requested_mode{
      ThrottleControlMode::Disabled};
  std::atomic<ThrottleControlMode> telemetry_effective_mode{
      ThrottleControlMode::Disabled};
  std::atomic<float> telemetry_current_speed_kmh{0.0f};
  std::atomic<float> telemetry_feedforward_pedal_v{0.75f};
  std::atomic<float> telemetry_speed_error_kmh{0.0f};
  std::atomic<float> telemetry_integral_v{0.0f};
  std::atomic<float> telemetry_final_pedal_v{0.75f};
  std::atomic<float> telemetry_pedal_upper_v{3.45f};
  std::atomic<bool> telemetry_vehicle_speed_fresh{false};
  std::atomic<float> telemetry_vehicle_speed_age_ms{0.0f};
  std::atomic<std::uint64_t> telemetry_vehicle_speed_timestamp_ns{0};
  std::thread worker;
  LongitudinalControllerKind controller_kind = LongitudinalControllerKind::Keypad;
};

struct ThrottleVehicleSpeedRead {
  float current_speed_kmh = 0.0f;
  bool fresh = false;
  float age_ms = std::numeric_limits<float>::infinity();
  std::uint64_t timestamp_ns = 0;
};

ThrottleTxState& GetThrottleTxState() {
  static ThrottleTxState state;
  return state;
}

ThrottleVehicleSpeedRead ReadThrottleVehicleSpeed() {
  constexpr std::uint64_t kVehicleSpeedTimeoutNs = 100'000'000ULL;

  const std::uint64_t now_ns = TimeSyncNowNs();
  const LongitudinalVehicleSpeedSnapshot snapshot =
      ReadLongitudinalVehicleSpeed();
  const bool speed_fresh =
      IsLongitudinalVehicleSpeedFresh(snapshot, now_ns, kVehicleSpeedTimeoutNs);

  ThrottleVehicleSpeedRead output;
  output.current_speed_kmh = speed_fresh ? snapshot.speed_kmh : 0.0f;
  output.fresh = speed_fresh;
  output.timestamp_ns = snapshot.timestamp_ns;
  output.age_ms =
      snapshot.valid && now_ns >= snapshot.timestamp_ns
          ? static_cast<float>(now_ns - snapshot.timestamp_ns) / 1'000'000.0f
          : std::numeric_limits<float>::infinity();

  return output;
}

void StoreThrottleTelemetry(ThrottleTxState& state,
                            ThrottleControlMode requested_mode,
                            ThrottleControlMode effective_mode,
                            float target_speed_kmh,
                            float current_speed_kmh,
                            float feedforward_pedal_v,
                            float speed_error_kmh,
                            float integral_v,
                            float final_pedal_v,
                            float pedal_upper_v,
                            bool vehicle_speed_fresh,
                            float vehicle_speed_age_ms,
                            std::uint64_t vehicle_speed_timestamp_ns) {
  (void)target_speed_kmh;
  state.telemetry_requested_mode.store(requested_mode, std::memory_order_release);
  state.telemetry_effective_mode.store(effective_mode, std::memory_order_release);
  state.telemetry_current_speed_kmh.store(current_speed_kmh, std::memory_order_release);
  state.telemetry_feedforward_pedal_v.store(feedforward_pedal_v, std::memory_order_release);
  state.telemetry_speed_error_kmh.store(speed_error_kmh, std::memory_order_release);
  state.telemetry_integral_v.store(integral_v, std::memory_order_release);
  state.telemetry_final_pedal_v.store(final_pedal_v, std::memory_order_release);
  state.telemetry_pedal_upper_v.store(pedal_upper_v, std::memory_order_release);
  state.telemetry_vehicle_speed_fresh.store(vehicle_speed_fresh, std::memory_order_release);
  state.telemetry_vehicle_speed_age_ms.store(vehicle_speed_age_ms, std::memory_order_release);
  state.telemetry_vehicle_speed_timestamp_ns.store(vehicle_speed_timestamp_ns,
                                                  std::memory_order_release);
}

void StoreMinimumThrottleTelemetry(ThrottleTxState& state,
                                   ThrottleControlMode requested_mode,
                                   ThrottleControlMode effective_mode,
                                   float target_speed_kmh,
                                   float current_speed_kmh,
                                   float pedal_upper_v,
                                   bool vehicle_speed_fresh,
                                   float vehicle_speed_age_ms,
                                   std::uint64_t vehicle_speed_timestamp_ns) {
  StoreThrottleTelemetry(state,
                         requested_mode,
                         effective_mode,
                         target_speed_kmh,
                         current_speed_kmh,
                         0.75f,
                         0.0f,
                         0.0f,
                         0.75f,
                         pedal_upper_v,
                         vehicle_speed_fresh,
                         vehicle_speed_age_ms,
                         vehicle_speed_timestamp_ns);
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
    constexpr float kControlPeriodS = 0.020f;
    constexpr auto kControlPeriod = std::chrono::milliseconds(20);

    if (controller_kind == LongitudinalControllerKind::Pid) {
      SpeedPedalController throttle;
      ThrottleControlMode previous_mode = ThrottleControlMode::Disabled;
      float speed_hold_target_kmh = 0.0f;
      constexpr float kCoastIntegralRetention = 0.25f;
      constexpr float kSpeedHoldIntegralRetention = 0.15f;

      while (GetThrottleTxState().running.load(std::memory_order_acquire)) {
        ThrottleTxState& state = GetThrottleTxState();
        const ThrottleControlMode requested_mode =
            state.mode.load(std::memory_order_acquire);
        const float desired_speed_kmh =
            state.target_speed_kmh.load(std::memory_order_acquire);
        const ThrottleVehicleSpeedRead speed_read = ReadThrottleVehicleSpeed();
        const float current_speed_kmh = speed_read.current_speed_kmh;
        const ThrottleControlMode effective_mode =
            speed_read.fresh ? requested_mode : ThrottleControlMode::Disabled;

        if (effective_mode != previous_mode) {
          switch (effective_mode) {
            case ThrottleControlMode::Disabled:
              throttle.Reset();
              speed_hold_target_kmh = 0.0f;
              break;
            case ThrottleControlMode::Coast:
              throttle.PrepareForCoast(kCoastIntegralRetention);
              break;
            case ThrottleControlMode::SpeedHold:
              speed_hold_target_kmh = current_speed_kmh;
              throttle.PrepareForSpeedHold(kSpeedHoldIntegralRetention);
              break;
            case ThrottleControlMode::SpeedTracking:
              if (previous_mode == ThrottleControlMode::Disabled) {
                throttle.Reset();
              }
              break;
          }
          previous_mode = effective_mode;
        }

        float pedal_command_v = 0.75f;
        switch (effective_mode) {
          case ThrottleControlMode::SpeedHold:
            pedal_command_v = throttle.Compute(speed_hold_target_kmh,
                                               current_speed_kmh,
                                               kControlPeriodS);
            {
              const SpeedPedalControllerTelemetry telemetry =
                  throttle.LastTelemetry();
              StoreThrottleTelemetry(state,
                                     requested_mode,
                                     effective_mode,
                                     speed_hold_target_kmh,
                                     current_speed_kmh,
                                     telemetry.feedforward_pedal_v,
                                     telemetry.speed_error_kmh,
                                     telemetry.integral_v,
                                     telemetry.final_pedal_v,
                                     telemetry.pedal_upper_v,
                                     speed_read.fresh,
                                     speed_read.age_ms,
                                     speed_read.timestamp_ns);
            }
            break;
          case ThrottleControlMode::SpeedTracking:
            pedal_command_v = throttle.Compute(desired_speed_kmh,
                                               current_speed_kmh,
                                               kControlPeriodS);
            {
              const SpeedPedalControllerTelemetry telemetry =
                  throttle.LastTelemetry();
              StoreThrottleTelemetry(state,
                                     requested_mode,
                                     effective_mode,
                                     desired_speed_kmh,
                                     current_speed_kmh,
                                     telemetry.feedforward_pedal_v,
                                     telemetry.speed_error_kmh,
                                     telemetry.integral_v,
                                     telemetry.final_pedal_v,
                                     telemetry.pedal_upper_v,
                                     speed_read.fresh,
                                     speed_read.age_ms,
                                     speed_read.timestamp_ns);
            }
            break;
          case ThrottleControlMode::Coast:
          case ThrottleControlMode::Disabled:
            pedal_command_v = 0.75f;
            StoreMinimumThrottleTelemetry(state,
                                          requested_mode,
                                          effective_mode,
                                          desired_speed_kmh,
                                          current_speed_kmh,
                                          3.45f,
                                          speed_read.fresh,
                                          speed_read.age_ms,
                                          speed_read.timestamp_ns);
            break;
        }

        canbus_ctrl_pedal(pedal_command_v);
        std::this_thread::sleep_for(kControlPeriod);
      }
      canbus_ctrl_pedal(0.75);
      return;
    }

    while (GetThrottleTxState().running.load(std::memory_order_acquire)) {
      ThrottleTxState& state = GetThrottleTxState();
      const ThrottleControlMode requested_mode =
          state.mode.load(std::memory_order_acquire);
      const double desired_speed_kmh = std::max(
          0.0f,
          state.target_speed_kmh.load(std::memory_order_acquire));
      const ThrottleVehicleSpeedRead speed_read = ReadThrottleVehicleSpeed();
      const double current_speed_kmh = speed_read.current_speed_kmh;
      const ThrottleControlMode effective_mode =
          speed_read.fresh ? requested_mode : ThrottleControlMode::Disabled;

      double pedal_cmd = 0.75;
      if ((effective_mode == ThrottleControlMode::SpeedTracking ||
           effective_mode == ThrottleControlMode::SpeedHold) &&
          desired_speed_kmh > 0.2) {
        const double speed_error = desired_speed_kmh - current_speed_kmh;
        if (speed_error > 0.2) {
          pedal_cmd = 0.75 + std::clamp(speed_error * 2.56, 0.0, 2.05);
        }
      }
      StoreThrottleTelemetry(state,
                             requested_mode,
                             effective_mode,
                             static_cast<float>(desired_speed_kmh),
                             static_cast<float>(current_speed_kmh),
                             0.75f,
                             static_cast<float>(desired_speed_kmh - current_speed_kmh),
                             0.0f,
                             static_cast<float>(pedal_cmd),
                             3.45f,
                             speed_read.fresh,
                             speed_read.age_ms,
                             speed_read.timestamp_ns);

      canbus_ctrl_pedal(pedal_cmd);
      std::this_thread::sleep_for(kControlPeriod);
    }
    canbus_ctrl_pedal(0.75);
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
  SetThrottleControlRequest(0.0f, ThrottleControlMode::Disabled);
  canbus_ctrl_pedal(0.75);
  StopThrottleThread();
#else
  (void)active;
  (void)controller_kind;
#endif
}

void SetThrottleControlRequest(float target_speed_kmh, ThrottleControlMode mode) {
#ifdef CANBUS__
  ThrottleTxState& state = GetThrottleTxState();
  const float valid_target_speed_kmh =
      std::isfinite(target_speed_kmh) ? std::max(0.0f, target_speed_kmh) : 0.0f;
  state.target_speed_kmh.store(valid_target_speed_kmh, std::memory_order_release);
  state.mode.store(mode, std::memory_order_release);
#else
  (void)target_speed_kmh;
  (void)mode;
#endif
}

ThrottleControlTelemetry GetThrottleControlTelemetry() {
#ifdef CANBUS__
  ThrottleTxState& state = GetThrottleTxState();
  ThrottleControlTelemetry telemetry;
  telemetry.requested_mode =
      state.telemetry_requested_mode.load(std::memory_order_acquire);
  telemetry.effective_mode =
      state.telemetry_effective_mode.load(std::memory_order_acquire);
  telemetry.mode = telemetry.effective_mode;
  telemetry.target_speed_kmh =
      state.target_speed_kmh.load(std::memory_order_acquire);
  telemetry.current_speed_kmh =
      state.telemetry_current_speed_kmh.load(std::memory_order_acquire);
  telemetry.feedforward_pedal_v =
      state.telemetry_feedforward_pedal_v.load(std::memory_order_acquire);
  telemetry.speed_error_kmh =
      state.telemetry_speed_error_kmh.load(std::memory_order_acquire);
  telemetry.integral_v =
      state.telemetry_integral_v.load(std::memory_order_acquire);
  telemetry.final_pedal_v =
      state.telemetry_final_pedal_v.load(std::memory_order_acquire);
  telemetry.pedal_upper_v =
      state.telemetry_pedal_upper_v.load(std::memory_order_acquire);
  telemetry.vehicle_speed_fresh =
      state.telemetry_vehicle_speed_fresh.load(std::memory_order_acquire);
  telemetry.vehicle_speed_age_ms =
      state.telemetry_vehicle_speed_age_ms.load(std::memory_order_acquire);
  telemetry.vehicle_speed_timestamp_ns =
      state.telemetry_vehicle_speed_timestamp_ns.load(std::memory_order_acquire);
  return telemetry;
#else
  return ThrottleControlTelemetry{};
#endif
}

void StopThrottleRuntime() {
#ifdef CANBUS__
  SetThrottleControlRequest(0.0f, ThrottleControlMode::Disabled);
  StopThrottleThread();
#endif
}

}  // namespace controller
