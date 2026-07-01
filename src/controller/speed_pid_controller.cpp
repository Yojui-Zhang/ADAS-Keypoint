#include "speed_pid_controller.h"

#include <algorithm>

namespace controller {

double IncrementalSpeedPid::Compute(float target,
                                    float actual,
                                    const SpeedPidGains& gains) {
  const float e = target - actual;
  const float a = gains.kp + gains.ki + gains.kd;
  const float b = -2.0f * gains.kd - gains.kp;
  const float c = gains.kd;
  const float u_increment = a * e + b * e_pre_1_ + c * e_pre_2_;
  e_pre_2_ = e_pre_1_;
  e_pre_1_ = e;
  return u_increment;
}

void IncrementalSpeedPid::Reset() {
  e_pre_1_ = 0.0f;
  e_pre_2_ = 0.0f;
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

double ClampControllerValue(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

}  // namespace controller
