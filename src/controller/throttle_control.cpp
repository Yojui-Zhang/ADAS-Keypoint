#include "throttle_control.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
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

struct ThrottleRuntimeConfigState {
  std::mutex mutex;
  ThrottleRuntimeConfig config{};
};

ThrottleRuntimeConfigState& GetThrottleRuntimeConfigState() {
  static ThrottleRuntimeConfigState state;
  return state;
}

ThrottleRuntimeConfig ReadThrottleRuntimeConfig() {
  ThrottleRuntimeConfigState& state = GetThrottleRuntimeConfigState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.config;
}

#ifdef CANBUS__
struct ThrottleTxState {
  std::atomic<bool> running{false};
  std::atomic<std::uint64_t> request_sequence{0};
  std::atomic<float> target_speed_kmh{0.0f};
  std::atomic<float> requested_brake_0_10{0.0f};
  std::atomic<ThrottleControlMode> mode{ThrottleControlMode::Disabled};
  std::atomic<std::uint64_t> telemetry_sequence{0};
  std::atomic<ThrottleControlMode> telemetry_requested_mode{
      ThrottleControlMode::Disabled};
  std::atomic<ThrottleControlMode> telemetry_effective_mode{
      ThrottleControlMode::Disabled};
  std::atomic<float> telemetry_target_speed_kmh{0.0f};
  std::atomic<float> telemetry_current_speed_kmh{0.0f};
  std::atomic<float> telemetry_visible_target_speed_kmh{0.0f};
  std::atomic<float> telemetry_operating_speed_kmh{0.0f};
  std::atomic<float> telemetry_feedforward_pedal_v{0.75f};
  std::atomic<float> telemetry_speed_error_kmh{0.0f};
  std::atomic<float> telemetry_integral_v{0.0f};
  std::atomic<float> telemetry_desired_pedal_v{0.75f};
  std::atomic<float> telemetry_final_pedal_v{0.75f};
  std::atomic<float> telemetry_applied_pedal_v{0.75f};
  std::atomic<float> telemetry_pedal_upper_v{3.45f};
  std::atomic<float> telemetry_requested_brake_0_10{0.0f};
  std::atomic<bool> telemetry_brake_interlock_active{false};
  std::atomic<float> telemetry_measured_dt_s{0.0f};
  std::atomic<bool> telemetry_vehicle_speed_fresh{false};
  std::atomic<float> telemetry_vehicle_speed_age_ms{0.0f};
  std::atomic<std::uint64_t> telemetry_vehicle_speed_timestamp_ns{0};
  std::atomic<bool> telemetry_vehicle_acceleration_fresh{false};
  std::atomic<float> telemetry_raw_acceleration_mps2{0.0f};
  std::atomic<float> telemetry_filtered_acceleration_mps2{0.0f};
  std::atomic<float> telemetry_measured_jerk_mps3{0.0f};
  std::atomic<float> telemetry_allowed_acceleration_mps2{0.0f};
  std::atomic<bool> telemetry_acceleration_guard_active{false};
  std::atomic<bool> telemetry_jerk_guard_active{false};
  std::thread worker;
  LongitudinalControllerKind controller_kind = LongitudinalControllerKind::Keypad;
};

struct ThrottleVehicleSpeedRead {
  float current_speed_kmh = 0.0f;
  bool fresh = false;
  float age_ms = std::numeric_limits<float>::infinity();
  std::uint64_t timestamp_ns = 0;
};

struct ThrottleRequestSnapshot {
  float target_speed_kmh = 0.0f;
  float brake_0_10 = 0.0f;
  ThrottleControlMode mode = ThrottleControlMode::Disabled;
  bool valid = false;
};

ThrottleTxState& GetThrottleTxState() {
  static ThrottleTxState state;
  return state;
}

ThrottleRequestSnapshot ReadThrottleRequest(ThrottleTxState& state) noexcept {
  for (int attempt = 0; attempt < 8; ++attempt) {
    const std::uint64_t sequence_begin =
        state.request_sequence.load(std::memory_order_acquire);

    if ((sequence_begin & 1U) != 0U) {
      continue;
    }

    ThrottleRequestSnapshot snapshot;
    snapshot.target_speed_kmh =
        state.target_speed_kmh.load(std::memory_order_relaxed);
    snapshot.brake_0_10 =
        state.requested_brake_0_10.load(std::memory_order_relaxed);
    snapshot.mode = state.mode.load(std::memory_order_relaxed);

    const std::uint64_t sequence_end =
        state.request_sequence.load(std::memory_order_acquire);

    if (sequence_begin != sequence_end) {
      continue;
    }

    snapshot.valid = true;
    return snapshot;
  }

  ThrottleRequestSnapshot fallback;
  fallback.brake_0_10 = 10.0f;
  fallback.mode = ThrottleControlMode::Disabled;
  return fallback;
}

ThrottleVehicleSpeedRead ReadThrottleVehicleSpeed(
    const std::uint64_t timeout_ns) {
  const std::uint64_t now_ns = TimeSyncNowNs();
  const LongitudinalVehicleSpeedSnapshot snapshot =
      ReadLongitudinalVehicleSpeed();
  const bool speed_fresh =
      IsLongitudinalVehicleSpeedFresh(snapshot, now_ns, timeout_ns);

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

class ThrottleModeTransitionFilter {
public:
  explicit ThrottleModeTransitionFilter(
      ThrottleModeTransitionConfig config = {}) noexcept
      : config_(config) {}

  ThrottleControlMode Update(ThrottleControlMode requested_mode,
                             float dt_s) noexcept {
    const float valid_dt_s =
        std::isfinite(dt_s) ? std::clamp(dt_s, 0.0f, 0.100f) : 0.0f;

    if (requested_mode == ThrottleControlMode::Disabled) {
      coast_request_duration_s_ = 0.0f;
      effective_mode_ = ThrottleControlMode::Disabled;
      return effective_mode_;
    }

    if (requested_mode != ThrottleControlMode::Coast) {
      coast_request_duration_s_ = 0.0f;
      effective_mode_ = requested_mode;
      return effective_mode_;
    }

    if (effective_mode_ == ThrottleControlMode::Coast) {
      return effective_mode_;
    }

    coast_request_duration_s_ += valid_dt_s;
    if (coast_request_duration_s_ >= config_.coast_entry_delay_s) {
      effective_mode_ = ThrottleControlMode::Coast;
    }

    return effective_mode_;
  }

private:
  ThrottleModeTransitionConfig config_{};
  ThrottleControlMode effective_mode_ = ThrottleControlMode::Disabled;
  float coast_request_duration_s_ = 0.0f;
};

struct PedalSafetyOutput {
  float pedal_v = 0.75f;
  float filtered_acceleration_mps2 = 0.0f;
  float measured_jerk_mps3 = 0.0f;
  float allowed_acceleration_mps2 = 0.0f;
  bool acceleration_guard_active = false;
  bool jerk_guard_active = false;
};

class LongitudinalPedalSafetyGuard {
public:
  explicit LongitudinalPedalSafetyGuard(
      LongitudinalPedalSafetyConfig config = {},
      float pedal_min_v = 0.75f,
      float pedal_max_v = 3.45f) noexcept
      : config_(config),
        pedal_min_v_(pedal_min_v),
        pedal_max_v_(pedal_max_v) {}

  PedalSafetyOutput Update(float requested_pedal_v,
                           float previous_applied_pedal_v,
                           float current_speed_kmh,
                           float measured_acceleration_mps2,
                           bool acceleration_fresh,
                           float dt_s) noexcept {
    PedalSafetyOutput output;
    output.pedal_v =
        std::clamp(requested_pedal_v, pedal_min_v_, pedal_max_v_);

    const float valid_dt_s =
        std::clamp(std::isfinite(dt_s) ? dt_s : 0.020f, 0.005f, 0.100f);

    if (!acceleration_fresh || !std::isfinite(measured_acceleration_mps2)) {
      return output;
    }

    if (!initialized_) {
      filtered_acceleration_mps2_ = measured_acceleration_mps2;
      previous_filtered_acceleration_mps2_ = filtered_acceleration_mps2_;
      initialized_ = true;
    }

    const float alpha =
        valid_dt_s / (config_.acceleration_filter_tau_s + valid_dt_s);
    previous_filtered_acceleration_mps2_ = filtered_acceleration_mps2_;
    filtered_acceleration_mps2_ +=
        alpha * (measured_acceleration_mps2 - filtered_acceleration_mps2_);

    const float jerk_mps3 =
        (filtered_acceleration_mps2_ - previous_filtered_acceleration_mps2_) /
        valid_dt_s;

    const float speed_ratio =
        std::clamp(std::max(0.0f, current_speed_kmh) /
                       config_.high_speed_transition_kmh,
                   0.0f,
                   1.0f);
    const float allowed_acceleration_mps2 =
        config_.low_speed_accel_limit_mps2 +
        speed_ratio *
            (config_.high_speed_accel_limit_mps2 -
             config_.low_speed_accel_limit_mps2);

    if (jerk_mps3 > config_.maximum_positive_jerk_mps3 &&
        output.pedal_v > previous_applied_pedal_v) {
      output.pedal_v = previous_applied_pedal_v;
      output.jerk_guard_active = true;
    }

    if (filtered_acceleration_mps2_ > allowed_acceleration_mps2) {
      output.pedal_v = std::min(output.pedal_v, previous_applied_pedal_v);
      output.acceleration_guard_active = true;
    }

    if (filtered_acceleration_mps2_ >
        allowed_acceleration_mps2 +
            config_.hard_acceleration_margin_mps2) {
      output.pedal_v =
          std::min(output.pedal_v,
                   std::max(pedal_min_v_,
                            previous_applied_pedal_v -
                                config_.hard_release_rate_v_per_s *
                                    valid_dt_s));
      output.acceleration_guard_active = true;
    }

    output.filtered_acceleration_mps2 = filtered_acceleration_mps2_;
    output.measured_jerk_mps3 = jerk_mps3;
    output.allowed_acceleration_mps2 = allowed_acceleration_mps2;

    return output;
  }

  void Reset() noexcept {
    initialized_ = false;
    filtered_acceleration_mps2_ = 0.0f;
    previous_filtered_acceleration_mps2_ = 0.0f;
  }

private:
  LongitudinalPedalSafetyConfig config_{};
  float pedal_min_v_ = 0.75f;
  float pedal_max_v_ = 3.45f;
  bool initialized_ = false;
  float filtered_acceleration_mps2_ = 0.0f;
  float previous_filtered_acceleration_mps2_ = 0.0f;
};

void StoreThrottleTelemetry(ThrottleTxState& state,
                            ThrottleControlMode requested_mode,
                            ThrottleControlMode effective_mode,
                            float target_speed_kmh,
                            float current_speed_kmh,
                            float visible_target_speed_kmh,
                            float operating_speed_kmh,
                            float feedforward_pedal_v,
                            float speed_error_kmh,
                            float integral_v,
                            float desired_pedal_v,
                            float final_pedal_v,
                            float applied_pedal_v,
                            float pedal_upper_v,
                            float requested_brake_0_10,
                            bool brake_interlock_active,
                            float measured_dt_s,
                            bool vehicle_speed_fresh,
                            float vehicle_speed_age_ms,
                            std::uint64_t vehicle_speed_timestamp_ns,
                            bool vehicle_acceleration_fresh,
                            float raw_acceleration_mps2,
                            float filtered_acceleration_mps2,
                            float measured_jerk_mps3,
                            float allowed_acceleration_mps2,
                            bool acceleration_guard_active,
                            bool jerk_guard_active) {
  state.telemetry_sequence.fetch_add(1, std::memory_order_acq_rel);
  state.telemetry_requested_mode.store(requested_mode, std::memory_order_relaxed);
  state.telemetry_effective_mode.store(effective_mode, std::memory_order_relaxed);
  state.telemetry_target_speed_kmh.store(target_speed_kmh, std::memory_order_relaxed);
  state.telemetry_current_speed_kmh.store(current_speed_kmh, std::memory_order_relaxed);
  state.telemetry_visible_target_speed_kmh.store(visible_target_speed_kmh,
                                                std::memory_order_relaxed);
  state.telemetry_operating_speed_kmh.store(operating_speed_kmh,
                                           std::memory_order_relaxed);
  state.telemetry_feedforward_pedal_v.store(feedforward_pedal_v, std::memory_order_relaxed);
  state.telemetry_speed_error_kmh.store(speed_error_kmh, std::memory_order_relaxed);
  state.telemetry_integral_v.store(integral_v, std::memory_order_relaxed);
  state.telemetry_desired_pedal_v.store(desired_pedal_v, std::memory_order_relaxed);
  state.telemetry_final_pedal_v.store(final_pedal_v, std::memory_order_relaxed);
  state.telemetry_applied_pedal_v.store(applied_pedal_v, std::memory_order_relaxed);
  state.telemetry_pedal_upper_v.store(pedal_upper_v, std::memory_order_relaxed);
  state.telemetry_requested_brake_0_10.store(requested_brake_0_10,
                                            std::memory_order_relaxed);
  state.telemetry_brake_interlock_active.store(brake_interlock_active,
                                               std::memory_order_relaxed);
  state.telemetry_measured_dt_s.store(measured_dt_s, std::memory_order_relaxed);
  state.telemetry_vehicle_speed_fresh.store(vehicle_speed_fresh, std::memory_order_relaxed);
  state.telemetry_vehicle_speed_age_ms.store(vehicle_speed_age_ms, std::memory_order_relaxed);
  state.telemetry_vehicle_speed_timestamp_ns.store(vehicle_speed_timestamp_ns,
                                                  std::memory_order_relaxed);
  state.telemetry_vehicle_acceleration_fresh.store(vehicle_acceleration_fresh,
                                                   std::memory_order_relaxed);
  state.telemetry_raw_acceleration_mps2.store(raw_acceleration_mps2,
                                             std::memory_order_relaxed);
  state.telemetry_filtered_acceleration_mps2.store(filtered_acceleration_mps2,
                                                  std::memory_order_relaxed);
  state.telemetry_measured_jerk_mps3.store(measured_jerk_mps3,
                                          std::memory_order_relaxed);
  state.telemetry_allowed_acceleration_mps2.store(allowed_acceleration_mps2,
                                                 std::memory_order_relaxed);
  state.telemetry_acceleration_guard_active.store(acceleration_guard_active,
                                                  std::memory_order_relaxed);
  state.telemetry_jerk_guard_active.store(jerk_guard_active,
                                          std::memory_order_relaxed);
  state.telemetry_sequence.fetch_add(1, std::memory_order_release);
}

void StoreMinimumThrottleTelemetry(ThrottleTxState& state,
                                   ThrottleControlMode requested_mode,
                                   ThrottleControlMode effective_mode,
                                   float target_speed_kmh,
                                   float current_speed_kmh,
                                   float pedal_upper_v,
                                   float requested_brake_0_10,
                                   bool brake_interlock_active,
                                   float measured_dt_s,
                                   bool vehicle_speed_fresh,
                                   float vehicle_speed_age_ms,
                                   std::uint64_t vehicle_speed_timestamp_ns,
                                   bool vehicle_acceleration_fresh,
                                   float raw_acceleration_mps2) {
  StoreThrottleTelemetry(state,
                         requested_mode,
                         effective_mode,
                         target_speed_kmh,
                         current_speed_kmh,
                         current_speed_kmh,
                         current_speed_kmh,
                         0.75f,
                         0.0f,
                         0.0f,
                         0.75f,
                         0.75f,
                         0.75f,
                         pedal_upper_v,
                         requested_brake_0_10,
                         brake_interlock_active,
                         measured_dt_s,
                         vehicle_speed_fresh,
                         vehicle_speed_age_ms,
                         vehicle_speed_timestamp_ns,
                         vehicle_acceleration_fresh,
                         raw_acceleration_mps2,
                         0.0f,
                         0.0f,
                         0.0f,
                         false,
                         false);
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

  const ThrottleRuntimeConfig runtime_config = ReadThrottleRuntimeConfig();
  state.controller_kind = controller_kind;
  state.running.store(true, std::memory_order_release);
  state.worker = std::thread([controller_kind, runtime_config]() {
    const auto kControlPeriod =
        std::chrono::milliseconds(runtime_config.control_period_ms);
    const std::uint64_t vehicle_speed_timeout_ns =
        runtime_config.vehicle_speed_timeout_ms * 1'000'000ULL;
    const std::uint64_t acceleration_timeout_ns =
        runtime_config.acceleration_timeout_ms * 1'000'000ULL;

    if (controller_kind == LongitudinalControllerKind::Pid) {
      SpeedPedalController throttle(runtime_config.speed_pid);
      ThrottleModeTransitionFilter mode_filter(runtime_config.mode_transition);
      LongitudinalPedalSafetyGuard pedal_safety_guard(
          runtime_config.safety,
          runtime_config.speed_pid.pedal_min_v,
          runtime_config.speed_pid.pedal_hard_max_v);

      ThrottleControlMode previous_mode = ThrottleControlMode::Disabled;

      float speed_hold_target_kmh = 0.0f;
      float last_applied_pedal_v = 0.75f;

      auto previous_tick = std::chrono::steady_clock::now();
      auto next_wakeup = previous_tick + kControlPeriod;

      while (GetThrottleTxState().running.load(std::memory_order_acquire)) {
        const auto current_tick = std::chrono::steady_clock::now();
        const float measured_dt_s =
            std::chrono::duration<float>(current_tick - previous_tick).count();
        previous_tick = current_tick;

        ThrottleTxState& state = GetThrottleTxState();
        const ThrottleRequestSnapshot request = ReadThrottleRequest(state);
        const float requested_brake_0_10 =
            request.valid ? request.brake_0_10 : 10.0f;
        const bool brake_interlock_active =
            !request.valid ||
            requested_brake_0_10 >
                runtime_config.brake_interlock_threshold_0_10;
        const ThrottleControlMode requested_mode =
            request.valid ? request.mode : ThrottleControlMode::Disabled;
        const float desired_speed_kmh =
            request.valid ? request.target_speed_kmh : 0.0f;
        const ThrottleVehicleSpeedRead speed_read =
            ReadThrottleVehicleSpeed(vehicle_speed_timeout_ns);
        const float current_speed_kmh = speed_read.current_speed_kmh;
        const ThrottleControlMode interlocked_requested_mode =
            brake_interlock_active ? ThrottleControlMode::Disabled
                                   : requested_mode;
        const ThrottleControlMode valid_requested_mode =
            speed_read.fresh ? interlocked_requested_mode
                             : ThrottleControlMode::Disabled;
        const ThrottleControlMode effective_mode =
            mode_filter.Update(valid_requested_mode, measured_dt_s);

        const std::uint64_t now_ns = TimeSyncNowNs();
        const LongitudinalVehicleAccelerationSnapshot acceleration_snapshot =
            ReadLongitudinalVehicleAcceleration();
        const bool acceleration_fresh =
            IsLongitudinalVehicleAccelerationFresh(acceleration_snapshot,
                                                   now_ns,
                                                   acceleration_timeout_ns);

        if (effective_mode != previous_mode) {
          switch (effective_mode) {
            case ThrottleControlMode::Disabled:
              throttle.ForceIdle();
              speed_hold_target_kmh = 0.0f;
              break;
            case ThrottleControlMode::Coast:
              throttle.PrepareForCoast(
                  runtime_config.mode_transition.coast_integral_retention);
              break;
            case ThrottleControlMode::SpeedHold:
              speed_hold_target_kmh = current_speed_kmh;
              throttle.PrepareForSpeedHold(
                  runtime_config.mode_transition.speed_hold_integral_retention);
              break;
            case ThrottleControlMode::SpeedTracking:
              break;
          }
          previous_mode = effective_mode;
        }

        float pedal_command_v = 0.75f;
        float telemetry_target_speed_kmh = desired_speed_kmh;
        PedalSafetyOutput safety_output;

        switch (effective_mode) {
          case ThrottleControlMode::SpeedHold:
            telemetry_target_speed_kmh = speed_hold_target_kmh;
            pedal_command_v = throttle.Compute(speed_hold_target_kmh,
                                               current_speed_kmh,
                                               measured_dt_s);
            break;
          case ThrottleControlMode::SpeedTracking:
            pedal_command_v = throttle.Compute(desired_speed_kmh,
                                               current_speed_kmh,
                                               measured_dt_s);
            break;
          case ThrottleControlMode::Coast:
            pedal_command_v = throttle.ReleaseToIdle(measured_dt_s);
            break;
          case ThrottleControlMode::Disabled:
            throttle.ForceIdle();
            pedal_command_v = 0.75f;
            break;
        }

        if (effective_mode == ThrottleControlMode::Disabled ||
            brake_interlock_active) {
          throttle.ForceIdle();
          pedal_safety_guard.Reset();
          pedal_command_v = 0.75f;
        } else {
          safety_output =
              pedal_safety_guard.Update(pedal_command_v,
                                        last_applied_pedal_v,
                                        current_speed_kmh,
                                        acceleration_snapshot.acceleration_mps2,
                                        acceleration_fresh,
                                        measured_dt_s);
          pedal_command_v = safety_output.pedal_v;
          throttle.SynchronizeAppliedOutput(pedal_command_v);
        }

        last_applied_pedal_v = pedal_command_v;

        if (effective_mode == ThrottleControlMode::Disabled) {
          StoreMinimumThrottleTelemetry(state,
                                        requested_mode,
                                        effective_mode,
                                        telemetry_target_speed_kmh,
                                        current_speed_kmh,
                                        runtime_config.speed_pid.pedal_hard_max_v,
                                        requested_brake_0_10,
                                        brake_interlock_active,
                                        measured_dt_s,
                                        speed_read.fresh,
                                        speed_read.age_ms,
                                        speed_read.timestamp_ns,
                                        acceleration_fresh,
                                        acceleration_snapshot.acceleration_mps2);
        } else {
          const SpeedPedalControllerTelemetry telemetry =
              throttle.LastTelemetry();
          StoreThrottleTelemetry(state,
                                 requested_mode,
                                 effective_mode,
                                 telemetry_target_speed_kmh,
                                 current_speed_kmh,
                                 telemetry.visible_target_speed_kmh,
                                 telemetry.operating_speed_kmh,
                                 telemetry.feedforward_pedal_v,
                                 telemetry.speed_error_kmh,
                                 telemetry.integral_v,
                                 telemetry.desired_pedal_v,
                                 telemetry.final_pedal_v,
                                 pedal_command_v,
                                 telemetry.pedal_upper_v,
                                 requested_brake_0_10,
                                 brake_interlock_active,
                                 measured_dt_s,
                                 speed_read.fresh,
                                 speed_read.age_ms,
                                 speed_read.timestamp_ns,
                                 acceleration_fresh,
                                 acceleration_snapshot.acceleration_mps2,
                                 safety_output.filtered_acceleration_mps2,
                                 safety_output.measured_jerk_mps3,
                                 safety_output.allowed_acceleration_mps2,
                                 safety_output.acceleration_guard_active,
                                 safety_output.jerk_guard_active);
        }

        canbus_ctrl_pedal(pedal_command_v);
        std::this_thread::sleep_until(next_wakeup);
        next_wakeup += kControlPeriod;

        const auto after_sleep = std::chrono::steady_clock::now();
        if (next_wakeup < after_sleep) {
          next_wakeup = after_sleep + kControlPeriod;
        }
      }
      canbus_ctrl_pedal(0.75);
      return;
    }

    while (GetThrottleTxState().running.load(std::memory_order_acquire)) {
      ThrottleTxState& state = GetThrottleTxState();
      const float measured_dt_s =
          static_cast<float>(runtime_config.control_period_ms) / 1000.0f;
      const ThrottleRequestSnapshot request = ReadThrottleRequest(state);
      const float requested_brake_0_10 =
          request.valid ? request.brake_0_10 : 10.0f;
      const bool brake_interlock_active =
          !request.valid ||
          requested_brake_0_10 >
              runtime_config.brake_interlock_threshold_0_10;
      const ThrottleControlMode requested_mode =
          request.valid ? request.mode : ThrottleControlMode::Disabled;
      const double desired_speed_kmh =
          std::max(0.0f, request.valid ? request.target_speed_kmh : 0.0f);
      const ThrottleVehicleSpeedRead speed_read =
          ReadThrottleVehicleSpeed(vehicle_speed_timeout_ns);
      const double current_speed_kmh = speed_read.current_speed_kmh;
      const ThrottleControlMode effective_mode =
          speed_read.fresh && !brake_interlock_active
              ? requested_mode
              : ThrottleControlMode::Disabled;
      const std::uint64_t now_ns = TimeSyncNowNs();
      const LongitudinalVehicleAccelerationSnapshot acceleration_snapshot =
          ReadLongitudinalVehicleAcceleration();
      const bool acceleration_fresh =
          IsLongitudinalVehicleAccelerationFresh(acceleration_snapshot,
                                                 now_ns,
                                                 acceleration_timeout_ns);

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
                             static_cast<float>(desired_speed_kmh),
                             static_cast<float>(current_speed_kmh),
                             0.75f,
                             static_cast<float>(desired_speed_kmh - current_speed_kmh),
                             0.0f,
                             static_cast<float>(pedal_cmd),
                             static_cast<float>(pedal_cmd),
                             static_cast<float>(pedal_cmd),
                             runtime_config.speed_pid.pedal_hard_max_v,
                             requested_brake_0_10,
                             brake_interlock_active,
                             measured_dt_s,
                             speed_read.fresh,
                             speed_read.age_ms,
                             speed_read.timestamp_ns,
                             acceleration_fresh,
                             acceleration_snapshot.acceleration_mps2,
                             0.0f,
                             0.0f,
                             0.0f,
                             false,
                             false);

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
  SetThrottleControlRequest(0.0f, ThrottleControlMode::Disabled, 10.0f);
  canbus_ctrl_pedal(0.75);
  StopThrottleThread();
#else
  (void)active;
  (void)controller_kind;
#endif
}

void ConfigureThrottleRuntime(const ThrottleRuntimeConfig& config) {
  ThrottleRuntimeConfigState& state = GetThrottleRuntimeConfigState();

  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.config = config;
  }

  std::cout << "Loaded throttle calibration:\n"
            << config.calibration_id << '\n'
            << "profiles: " << config.speed_pid.profiles.size() << '\n'
            << "control period: " << config.control_period_ms << " ms\n"
            << "speed timeout: " << config.vehicle_speed_timeout_ms << " ms\n"
            << "acceleration timeout: " << config.acceleration_timeout_ms
            << " ms\n";
}

void SetThrottleControlRequest(float target_speed_kmh,
                               ThrottleControlMode requested_mode,
                               float brake_0_10) {
#ifdef CANBUS__
  ThrottleTxState& state = GetThrottleTxState();
  const ThrottleRuntimeConfig runtime_config = ReadThrottleRuntimeConfig();
  const float valid_target_speed_kmh =
      std::isfinite(target_speed_kmh) ? std::max(0.0f, target_speed_kmh) : 0.0f;
  const float valid_brake_0_10 =
      std::isfinite(brake_0_10) ? std::clamp(brake_0_10, 0.0f, 10.0f)
                                : 10.0f;
  const bool brake_active =
      valid_brake_0_10 >
      runtime_config.brake_interlock_threshold_0_10;
  const ThrottleControlMode safe_mode =
      brake_active ? ThrottleControlMode::Disabled : requested_mode;

  state.request_sequence.fetch_add(1, std::memory_order_acq_rel);
  state.target_speed_kmh.store(valid_target_speed_kmh,
                               std::memory_order_relaxed);
  state.requested_brake_0_10.store(valid_brake_0_10,
                                   std::memory_order_relaxed);
  state.mode.store(safe_mode, std::memory_order_relaxed);
  state.request_sequence.fetch_add(1, std::memory_order_release);

  if (brake_active) {
    canbus_ctrl_pedal(0.75);
  }
#else
  (void)target_speed_kmh;
  (void)requested_mode;
  (void)brake_0_10;
#endif
}

ThrottleControlTelemetry GetThrottleControlTelemetry() {
#ifdef CANBUS__
  ThrottleTxState& state = GetThrottleTxState();
  const ThrottleRuntimeConfig runtime_config = ReadThrottleRuntimeConfig();

  for (int attempt = 0; attempt < 8; ++attempt) {
    const std::uint64_t sequence_begin =
        state.telemetry_sequence.load(std::memory_order_acquire);

    if ((sequence_begin & 1U) != 0U) {
      continue;
    }

    ThrottleControlTelemetry telemetry;
    telemetry.requested_mode =
        state.telemetry_requested_mode.load(std::memory_order_relaxed);
    telemetry.effective_mode =
        state.telemetry_effective_mode.load(std::memory_order_relaxed);
    telemetry.mode = telemetry.effective_mode;
    telemetry.target_speed_kmh =
        state.telemetry_target_speed_kmh.load(std::memory_order_relaxed);
    telemetry.current_speed_kmh =
        state.telemetry_current_speed_kmh.load(std::memory_order_relaxed);
    telemetry.visible_target_speed_kmh =
        state.telemetry_visible_target_speed_kmh.load(std::memory_order_relaxed);
    telemetry.operating_speed_kmh =
        state.telemetry_operating_speed_kmh.load(std::memory_order_relaxed);
    telemetry.feedforward_pedal_v =
        state.telemetry_feedforward_pedal_v.load(std::memory_order_relaxed);
    telemetry.speed_error_kmh =
        state.telemetry_speed_error_kmh.load(std::memory_order_relaxed);
    telemetry.integral_v =
        state.telemetry_integral_v.load(std::memory_order_relaxed);
    telemetry.desired_pedal_v =
        state.telemetry_desired_pedal_v.load(std::memory_order_relaxed);
    telemetry.final_pedal_v =
        state.telemetry_final_pedal_v.load(std::memory_order_relaxed);
    telemetry.applied_pedal_v =
        state.telemetry_applied_pedal_v.load(std::memory_order_relaxed);
    telemetry.pedal_upper_v =
        state.telemetry_pedal_upper_v.load(std::memory_order_relaxed);
    telemetry.requested_brake_0_10 =
        state.telemetry_requested_brake_0_10.load(std::memory_order_relaxed);
    telemetry.brake_interlock_active =
        state.telemetry_brake_interlock_active.load(std::memory_order_relaxed);
    telemetry.measured_dt_s =
        state.telemetry_measured_dt_s.load(std::memory_order_relaxed);
    telemetry.vehicle_speed_fresh =
        state.telemetry_vehicle_speed_fresh.load(std::memory_order_relaxed);
    telemetry.vehicle_speed_age_ms =
        state.telemetry_vehicle_speed_age_ms.load(std::memory_order_relaxed);
    telemetry.vehicle_speed_timestamp_ns =
        state.telemetry_vehicle_speed_timestamp_ns.load(std::memory_order_relaxed);
    telemetry.vehicle_acceleration_fresh =
        state.telemetry_vehicle_acceleration_fresh.load(std::memory_order_relaxed);
    telemetry.raw_acceleration_mps2 =
        state.telemetry_raw_acceleration_mps2.load(std::memory_order_relaxed);
    telemetry.filtered_acceleration_mps2 =
        state.telemetry_filtered_acceleration_mps2.load(std::memory_order_relaxed);
    telemetry.measured_jerk_mps3 =
        state.telemetry_measured_jerk_mps3.load(std::memory_order_relaxed);
    telemetry.allowed_acceleration_mps2 =
        state.telemetry_allowed_acceleration_mps2.load(std::memory_order_relaxed);
    telemetry.acceleration_guard_active =
        state.telemetry_acceleration_guard_active.load(std::memory_order_relaxed);
    telemetry.jerk_guard_active =
        state.telemetry_jerk_guard_active.load(std::memory_order_relaxed);
    telemetry.calibration_id = runtime_config.calibration_id;

    const std::uint64_t sequence_end =
        state.telemetry_sequence.load(std::memory_order_acquire);

    if (sequence_begin == sequence_end) {
      return telemetry;
    }
  }

  ThrottleControlTelemetry fail_safe;
  fail_safe.mode = ThrottleControlMode::Disabled;
  fail_safe.requested_mode = ThrottleControlMode::Disabled;
  fail_safe.effective_mode = ThrottleControlMode::Disabled;
  fail_safe.final_pedal_v = 0.75f;
  fail_safe.applied_pedal_v = 0.75f;
  fail_safe.calibration_id = runtime_config.calibration_id;
  return fail_safe;
#else
  return ThrottleControlTelemetry{};
#endif
}

void StopThrottleRuntime() {
#ifdef CANBUS__
  SetThrottleControlRequest(0.0f, ThrottleControlMode::Disabled, 10.0f);
  StopThrottleThread();
#endif
}

}  // namespace controller
