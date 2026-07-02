#include "speed_pid_controller.h"

#include <algorithm>
#include <array>

namespace controller {
namespace {

struct SpeedPidProfile {
  double speed_kmh;
  SpeedPidGains gains;
  double pedal_upper_limit;
  float max_visible_error_kmh;
};

// The current velocity-form PID settles to roughly ki * speed_error for a
// constant error, so ki is scheduled as pedal-command sensitivity per km/h.
constexpr std::array<SpeedPidProfile, 11> kSpeedPidProfiles = {{
    {0.0,   {0.70f, 1.10f, 0.18f}, 1.40, 6.0f},
    {10.0,  {0.82f, 1.30f, 0.20f}, 1.55, 7.0f},
    {20.0,  {0.98f, 1.60f, 0.22f}, 1.80, 8.5f},
    {30.0,  {1.20f, 2.05f, 0.25f}, 2.15, 10.0f},
    {40.0,  {1.48f, 2.65f, 0.30f}, 2.60, 12.0f},
    {50.0,  {1.82f, 3.35f, 0.36f}, 3.05, 14.0f},
    {60.0,  {2.22f, 4.15f, 0.42f}, 3.45, 16.0f},
    {70.0,  {2.68f, 5.05f, 0.48f}, 3.85, 18.0f},
    {80.0,  {3.12f, 5.85f, 0.52f}, 4.15, 19.0f},
    {90.0,  {3.48f, 6.55f, 0.56f}, 4.40, 20.0f},
    {100.0, {3.82f, 7.20f, 0.60f}, 4.60, 20.0f},
}};

float LerpFloat(float a, float b, double t) {
  return static_cast<float>(static_cast<double>(a) +
                            (static_cast<double>(b) - static_cast<double>(a)) * t);
}

double LerpDouble(double a, double b, double t) {
  return a + (b - a) * t;
}

SpeedPidProfile InterpolateSpeedPidProfile(double speed_kmh) {
  if (speed_kmh <= kSpeedPidProfiles.front().speed_kmh) {
    return kSpeedPidProfiles.front();
  }
  if (speed_kmh >= kSpeedPidProfiles.back().speed_kmh) {
    return kSpeedPidProfiles.back();
  }

  for (std::size_t i = 1; i < kSpeedPidProfiles.size(); ++i) {
    const SpeedPidProfile& hi = kSpeedPidProfiles[i];
    if (speed_kmh > hi.speed_kmh) {
      continue;
    }

    const SpeedPidProfile& lo = kSpeedPidProfiles[i - 1];
    const double span = std::max(1e-6, hi.speed_kmh - lo.speed_kmh);
    const double t = (speed_kmh - lo.speed_kmh) / span;

    SpeedPidProfile out = lo;
    out.speed_kmh = speed_kmh;
    out.gains.kp = LerpFloat(lo.gains.kp, hi.gains.kp, t);
    out.gains.ki = LerpFloat(lo.gains.ki, hi.gains.ki, t);
    out.gains.kd = LerpFloat(lo.gains.kd, hi.gains.kd, t);
    out.pedal_upper_limit =
        LerpDouble(lo.pedal_upper_limit, hi.pedal_upper_limit, t);
    out.max_visible_error_kmh =
        LerpFloat(lo.max_visible_error_kmh, hi.max_visible_error_kmh, t);
    return out;
  }

  return kSpeedPidProfiles.back();
}

}  // namespace

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
  return InterpolateSpeedPidProfile(speed_kmh).gains;
}

double SelectSpeedPidPedalUpperLimit(double speed_kmh) {
  return InterpolateSpeedPidProfile(speed_kmh).pedal_upper_limit;
}

float LimitSpeedPidTarget(float desired_speed_kmh, float current_speed_kmh) {
  const float speed_error_kmh = desired_speed_kmh - current_speed_kmh;
  if (speed_error_kmh <= 0.0f) {
    return desired_speed_kmh;
  }

  const float max_visible_error_kmh =
      InterpolateSpeedPidProfile(current_speed_kmh).max_visible_error_kmh;

  return current_speed_kmh + std::min(speed_error_kmh, max_visible_error_kmh);
}

double ClampControllerValue(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

}  // namespace controller
