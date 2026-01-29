#pragma once
#include <cmath>
#include <algorithm>

namespace stability {

static constexpr double kPi = 3.14159265358979323846;

inline double Clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}

inline double Deg2Rad(double deg) { return deg * kPi / 180.0; }
inline double Rad2Deg(double rad) { return rad * 180.0 / kPi; }

inline bool IsFinite(double v) { return std::isfinite(v); }

// 由「路輪轉角 delta(rad)」與軸距 L 得到曲率 kappa(1/m)
// kappa = tan(delta) / L
inline double CurvatureFromSteerRad(double delta_road_rad, double wheelbase_m) {
  const double L = std::max(1e-3, wheelbase_m);
  return std::tan(delta_road_rad) / L;
}

// 側向加速度 a_lat = v^2 * |kappa|
inline double LateralAccel(double v_mps, double kappa_abs) {
  const double v = std::max(0.0, v_mps);
  return v * v * std::max(0.0, kappa_abs);
}

// 向心力 F_c = m * a_lat
inline double CentripetalForce(double mass_kg, double a_lat_mps2) {
  return std::max(0.0, mass_kg) * std::max(0.0, a_lat_mps2);
}

// 動能 KE = 0.5*m*v^2
inline double KineticEnergy(double mass_kg, double v_mps) {
  const double v = std::max(0.0, v_mps);
  return 0.5 * std::max(0.0, mass_kg) * v * v;
}

} // namespace stability

