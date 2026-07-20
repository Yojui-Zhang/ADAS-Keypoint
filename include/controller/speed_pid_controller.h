#pragma once

#include <cstdint>

namespace controller {

struct SpeedPedalProfile {
  float speed_kmh = 0.0f;
  float feedforward_pedal_v = 0.75f;
  float kp_v_per_kmh = 0.0f;
  float ki_v_per_kmh_s = 0.0f;
  float pedal_upper_v = 3.45f;
  float max_visible_error_kmh = 10.0f;
};

struct SpeedPedalControllerConfig {
  float pedal_min_v = 0.75f;
  float pedal_hard_max_v = 3.45f;
  float speed_error_deadband_kmh = 0.30f;
  float integral_min_v = -0.40f;
  float integral_max_v = 0.35f;
  float pedal_rise_rate_v_per_s = 1.60f;
  float pedal_fall_rate_v_per_s = 3.00f;
  float min_dt_s = 0.005f;
  float max_dt_s = 0.100f;
};

struct SpeedPedalControllerTelemetry {
  float target_speed_kmh = 0.0f;
  float actual_speed_kmh = 0.0f;
  float visible_target_speed_kmh = 0.0f;
  float speed_error_kmh = 0.0f;
  float feedforward_pedal_v = 0.75f;
  float proportional_v = 0.0f;
  float integral_v = 0.0f;
  float final_pedal_v = 0.75f;
  float pedal_upper_v = 3.45f;
};

class SpeedPedalController {
public:
  explicit SpeedPedalController(SpeedPedalControllerConfig config = {});

  float Compute(float target_speed_kmh,
                float actual_speed_kmh,
                float dt_s);

  void Reset() noexcept;

  void PrepareForCoast(float integral_retention) noexcept;

  void PrepareForSpeedHold(float integral_retention) noexcept;

  [[nodiscard]] float LastOutputV() const noexcept;

  [[nodiscard]] SpeedPedalControllerTelemetry LastTelemetry() const noexcept;

private:
  static float Clamp(float value, float lower, float upper) noexcept;

  SpeedPedalControllerConfig config_{};
  SpeedPedalControllerTelemetry telemetry_{};
  float integral_v_ = 0.0f;
  float last_output_v_ = 0.75f;
};

SpeedPedalProfile SelectSpeedPedalProfile(float target_speed_kmh);

float LimitSpeedPidTarget(float desired_speed_kmh,
                          float current_speed_kmh);

}  // namespace controller
