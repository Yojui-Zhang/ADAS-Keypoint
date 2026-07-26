#pragma once

#include "longitudinal_actuation_config.h"

namespace controller {

struct SpeedPedalControllerTelemetry {
  float target_speed_kmh = 0.0f;
  float actual_speed_kmh = 0.0f;
  float visible_target_speed_kmh = 0.0f;
  float operating_speed_kmh = 0.0f;

  float speed_error_kmh = 0.0f;
  float feedforward_pedal_v = 0.75f;
  float proportional_v = 0.0f;
  float integral_v = 0.0f;

  float desired_pedal_v = 0.75f;
  float final_pedal_v = 0.75f;
  float pedal_upper_v = 3.45f;
};

class SpeedPedalController {
public:
  explicit SpeedPedalController(SpeedPedalControllerConfig config = {});

  float Compute(float target_speed_kmh,
                float actual_speed_kmh,
                float dt_s);

  float ReleaseToIdle(float dt_s) noexcept;

  void ForceIdle() noexcept;

  void Reset() noexcept;

  void PrepareForCoast(float integral_retention) noexcept;

  void PrepareForSpeedHold(float integral_retention) noexcept;

  void SynchronizeAppliedOutput(float applied_pedal_v) noexcept;

  [[nodiscard]] float LastOutputV() const noexcept;

  [[nodiscard]] SpeedPedalControllerTelemetry LastTelemetry() const noexcept;

private:
  static float Clamp(float value, float lower, float upper) noexcept;

  float SlewToward(float desired_pedal_v,
                   float rise_rate_v_per_s,
                   float fall_rate_v_per_s,
                   float dt_s) const noexcept;

  [[nodiscard]] SpeedPedalProfile SelectProfile(
      float operating_speed_kmh) const;

  [[nodiscard]] float LimitTargetSpeed(float desired_speed_kmh,
                                       float actual_speed_kmh) const;

  SpeedPedalControllerConfig config_{};
  SpeedPedalControllerTelemetry telemetry_{};
  float integral_v_ = 0.0f;
  float last_output_v_ = 0.75f;
};

SpeedPedalProfile SelectSpeedPedalProfile(float operating_speed_kmh);

float LimitSpeedPidTarget(float desired_speed_kmh,
                          float current_speed_kmh);

}  // namespace controller
