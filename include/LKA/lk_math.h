#pragma once

#include <cmath>
#include <algorithm>

namespace lane_keeping {
namespace internal {

constexpr double kPi = 3.14159265358979323846;

inline double Rad2Deg(double rad) { return rad * 180.0 / kPi; }
inline double Deg2Rad(double deg) { return deg * kPi / 180.0; }

inline double Clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// Angle normalization to [-pi, pi]
inline double WrapPi(double rad) {
    while (rad >  kPi) rad -= 2.0 * kPi;
    while (rad < -kPi) rad += 2.0 * kPi;
    return rad;
}

// Steering rate limiter (rad/s). Keeps behavior identical to the original implementation.
inline double RateLimitRad(double target_rad, double last_rad, double max_rate_deg_s, double dt_s) {
    if (dt_s <= 1e-6) return target_rad;

    const double max_rate_rad_s = Deg2Rad(std::max(0.0, max_rate_deg_s));
    const double max_delta = max_rate_rad_s * dt_s;

    const double diff = target_rad - last_rad;
    if (diff >  max_delta) return last_rad + max_delta;
    if (diff < -max_delta) return last_rad - max_delta;
    return target_rad;
}

} // namespace internal
} // namespace lane_keeping
