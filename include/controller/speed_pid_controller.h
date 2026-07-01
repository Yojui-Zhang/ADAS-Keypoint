#pragma once

namespace controller {

struct SpeedPidGains {
  float kp = 5.4f;
  float ki = 30.0f;
  float kd = 2.5f;
};

class IncrementalSpeedPid {
public:
  double Compute(float target, float actual, const SpeedPidGains& gains);
  void Reset();

private:
  float e_pre_1_ = 0.0f;
  float e_pre_2_ = 0.0f;
};

SpeedPidGains SelectSpeedPidGains(double speed_kmh);
double SelectSpeedPidPedalUpperLimit(double speed_kmh);
float LimitSpeedPidTarget(float desired_speed_kmh, float current_speed_kmh);
double ClampControllerValue(double x, double lo, double hi);

}  // namespace controller
