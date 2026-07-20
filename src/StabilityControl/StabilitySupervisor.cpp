#include "StabilitySupervisor.h"
#include "VehiclePhysics.h"

#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace {

struct Acc2 { double alat; double along; };

static inline double sqr(double x) { return x * x; }

static inline double clampd(double x, double lo, double hi) {
  return std::max(lo, std::min(x, hi));
}

// 解析式「加速度域」投影：S = { (alat, along) | alat^2+along^2 <= A^2, |alat| <= alat_max }
// 以 weighted distance 最小化近似：min w_lat(alat-a0)^2 + w_long(along-b0)^2
static Acc2 ProjectToFeasibleSet(const Acc2& u,
                                double A,
                                double alat_max,
                                double w_lat,
                                double w_long)
{
  Acc2 x = u;

  // 1) hard strip
  if (std::fabs(x.alat) > alat_max) {
    x.alat = (x.alat > 0.0) ? +alat_max : -alat_max;
  }

  // 2) if inside disk, done
  const double r2 = sqr(x.alat) + sqr(x.along);
  const double A2 = sqr(A);
  if (r2 <= A2) return x;

  // Candidate A: keep alat, clamp along to circle
  Acc2 c1 = x;
  const double along_max = std::sqrt(std::max(0.0, A2 - sqr(c1.alat)));
  c1.along = clampd(c1.along, -along_max, +along_max);

  // Candidate B: radial projection to circle
  Acc2 c2 = x;
  const double r = std::sqrt(std::max(1e-12, r2));
  const double s = A / r;
  c2.alat  *= s;
  c2.along *= s;
  // radial 後 |alat| 只會變小，不會破 strip

  auto cost = [&](const Acc2& c) {
    return w_lat * sqr(c.alat - u.alat) + w_long * sqr(c.along - u.along);
  };

  return (cost(c1) <= cost(c2)) ? c1 : c2;
}

static inline double KmH2mps(double kmh) { return kmh / 3.6; }
static inline double mps2KmH(double mps) { return mps * 3.6; }

} // anonymous namespace

namespace stability {

VehicleControlCommand StabilitySupervisor::Update(double ego_speed_mps,
                                                  double dt_s,
                                                  const acc::AccCommand& acc_cmd,
                                                  double lka_steer_deg,
                                                  double yaw_rate_rps,
                                                  double alat_mps2,
                                                  const std::string& acc_dbg,
                                                  const std::string& lka_dbg)
{
  VehicleControlCommand out;

  dt_s = std::max(1e-3, dt_s);
  const double v = std::max(0.0, ego_speed_mps);

  // 0) sanity
  if (!IsFinite(v) || v > 120.0) {
    out.steer_deg  = 0.0f;
    out.speed_kmh  = 0.0f;
    out.brake_0_10 = 10.0f;
    out.debug      = "invalid ego speed -> full brake safe";
    return out;
  }
  if (!IsFinite(lka_steer_deg)) lka_steer_deg = last_steer_deg_;

  // 1) convert steer (wheel input) -> road wheel
  const double steer_ratio   = std::max(1e-3, cfg_.steering_ratio);
  const double delta_road_deg = lka_steer_deg / steer_ratio;
  const double delta_road_rad = Deg2Rad(delta_road_deg);

  // signed curvature & lateral accel (command-based)
  const double kappa_signed = CurvatureFromSteerRad(delta_road_rad, cfg_.wheelbase_m);
  const double kappa_abs    = std::fabs(kappa_signed);
  const double alat_nom     = v * v * kappa_signed;        // signed
  const double alat_nom_abs = std::fabs(alat_nom);

  // (for debug / slip)
  const double alat_demand_abs = LateralAccel(v, kappa_abs);
  const double F_c = CentripetalForce(cfg_.mass_kg, alat_demand_abs);
  const double KE  = KineticEnergy(cfg_.mass_kg, v);

  // 2) measured lateral accel (use IMU alat else yaw_rate*v)
  double alat_meas = std::numeric_limits<double>::quiet_NaN();
  if (std::isfinite(alat_mps2)) {
    alat_meas = alat_mps2;
  } else if (std::isfinite(yaw_rate_rps)) {
    alat_meas = yaw_rate_rps * v;
  }

  double alat_meas_abs = std::numeric_limits<double>::quiet_NaN();
  if (cfg_.use_measured_alat && std::isfinite(alat_meas)) {
    const double a = std::fabs(alat_meas);
    if (!alat_lpf_inited_) {
      alat_meas_lpf_mps2_ = a;
      alat_lpf_inited_ = true;
    } else {
      const double alpha = clampd(cfg_.alat_lpf_alpha, 0.0, 1.0);
      alat_meas_lpf_mps2_ = alpha * a + (1.0 - alpha) * alat_meas_lpf_mps2_;
    }
    alat_meas_abs = alat_meas_lpf_mps2_;
  }

  // 3) slip hysteresis (use abs demand vs mu_s*g)
  const double mu_s = std::max(0.05, cfg_.mu_static);
  const double mu_k = std::max(0.05, std::min(cfg_.mu_dynamic, mu_s));
  const double g    = std::max(1e-3, cfg_.g);

  const double alat_s_limit = mu_s * g;

  if (!in_slip_) {
    if (alat_demand_abs > alat_s_limit * cfg_.slip_enter_ratio) in_slip_ = true;
  } else {
    if (alat_demand_abs < alat_s_limit * cfg_.slip_exit_ratio) in_slip_ = false;
  }

  const double mu_target = in_slip_ ? mu_k : mu_s;
  const double a_mu = clampd(cfg_.mu_lowpass_alpha, 0.0, 0.999);
  mu_eff_ = a_mu * mu_eff_ + (1.0 - a_mu) * mu_target;

  // 4) friction circle budget (hard)
  double A = mu_eff_ * g * cfg_.total_safety; // total accel budget

  // 5) lateral comfort strip (hard in projection)
  const double alat_strip =
      std::min(cfg_.lat_accel_comfort_mps2, mu_eff_ * g * cfg_.lat_safety);

  // 6) curve speed limit: v <= sqrt(alat_strip / |kappa|)
  double v_curve_limit = 1e9;
  if (kappa_abs > 1e-6 && v > cfg_.min_speed_for_curvelimit_mps) {
    v_curve_limit = std::sqrt(std::max(0.0, alat_strip / kappa_abs));
  }

  // 7) parse ACC longitudinal need (speed/brake)
  const acc::AccConfig acc_cfg = acc::ACC_GetConfig();

  const double brake = clampd(acc_cmd.brake_0_10, 0.0, 10.0);
  const bool acc_idle_request =
      acc_cmd.longitudinal_phase == acc::AccLongitudinalPhase::Idle &&
      brake <= 1e-3 &&
      acc_cmd.speed_kmh <= 0.2f;
  const bool acc_coast_request =
      acc_cmd.longitudinal_phase == acc::AccLongitudinalPhase::Coasting &&
      brake <= 1e-3;
  const bool acc_zero_accel_request = acc_idle_request || acc_coast_request;

  const double v_acc_target = acc_zero_accel_request
      ? v
      : KmH2mps(std::max(0.0f, acc_cmd.speed_kmh));

  double a_brake_need = 0.0;
  if (brake > 1e-3) {
    const double multiplier = std::max(1e-3f, acc_cfg.brake_multiplier);
    const double full_decel_mps2 = std::max(1e-3f, acc_cfg.brake_full_decel_mps2);
    a_brake_need = (brake / multiplier) * full_decel_mps2;
  }

  double a_long_need = 0.0;
  if (a_brake_need > 1e-3) a_long_need = -a_brake_need;
  else if (acc_zero_accel_request) a_long_need = 0.0;
  else                     a_long_need = (v_acc_target - v) / dt_s;

  // 8) supervisor target speed (ACC vs curve), smoothing, rate limit (discrete-time)
  const double v_target_raw = std::min(v_acc_target, v_curve_limit);
  const bool curve_is_bottleneck = (v_curve_limit + 1e-3 < v_acc_target);
  double v_target = v_target_raw;

  if (acc_coast_request && !curve_is_bottleneck) {
    v_target = v;
    last_speed_cmd_mps_ = v;
  } else {
    const double speed_alpha = clampd(cfg_.speed_lowpass_alpha, 0.0, 0.999);
    if (last_speed_cmd_mps_ <= 1e-6) last_speed_cmd_mps_ = v_target;
    v_target = speed_alpha * last_speed_cmd_mps_ + (1.0 - speed_alpha) * v_target;

    const double dv_up = cfg_.max_speed_rise_mps2 * dt_s;
    const double dv_down = cfg_.max_speed_drop_mps2 * dt_s;
    v_target = clampd(v_target, last_speed_cmd_mps_ - dv_down, last_speed_cmd_mps_ + dv_up);
  }

  // If ACC is not asking for brake and the curve-speed limit is not the active
  // bottleneck, do not let target smoothing lag behind the current ego speed and
  // create a false decel request.
  const bool acc_requests_brake = (a_brake_need > 1e-3);
  const bool acc_wants_hold_or_accel = (v_acc_target + 1e-3 >= v);
  if (!acc_coast_request &&
      !acc_requests_brake &&
      !curve_is_bottleneck &&
      acc_wants_hold_or_accel) {
    v_target = std::max(v_target, v);
  }

  // nominal along from speed target
  double a_long_nom = (v_target - v) / dt_s;

  // if ACC explicitly wants more braking, respect it BEFORE projection (so projection really applies)
  if (a_long_need < a_long_nom) a_long_nom = a_long_need;

  // 9) projection in acceleration space (hard: disk ∩ strip)
  const double w_lat  = std::max(1e-6, cfg_.w_lat);
  const double w_long = std::max(1e-6, cfg_.w_long);

  Acc2 u_nom  { alat_nom, a_long_nom };
  Acc2 u_proj = ProjectToFeasibleSet(u_nom, A, alat_strip, w_lat, w_long);

  // 10) extra robustness: use measured a_lat (abs) as conservative "consumed lateral"
  //     => do not allow along exceed sqrt(A^2 - alat_used^2)
  double alat_used_abs = alat_nom_abs;
  if (std::isfinite(alat_meas_abs)) {
    alat_used_abs = std::max(alat_meas_abs, cfg_.alat_cmd_guard_ratio * alat_nom_abs);
  }
  alat_used_abs = std::min(alat_used_abs, A);
  const double along_left = std::sqrt(std::max(0.0, sqr(A) - sqr(alat_used_abs)));
  u_proj.along = clampd(u_proj.along, -along_left, +along_left);

  // 11) longitudinal comfort (soft, still cannot violate circle)
  double a_long_cmd = u_proj.along;

  const double a_long_accel_allow = std::min(along_left, cfg_.long_accel_comfort_mps2);
  double a_long_decel_allow       = std::min(along_left, cfg_.long_decel_comfort_mps2);

  if (std::isfinite(acc_cmd.TargetTTC) && acc_cmd.TargetTTC < cfg_.ttc_hard_guard_s) {
    a_long_decel_allow = std::min(along_left,
                                  std::min(cfg_.emergency_decel_cap_mps2, A));
  }

  if (a_long_cmd >= 0.0) a_long_cmd = std::min(a_long_cmd,  a_long_accel_allow);
  else                   a_long_cmd = std::max(a_long_cmd, -a_long_decel_allow);

  // 12) jerk limit (discrete-time), then re-project to guarantee circle
  const bool emergency = (std::isfinite(acc_cmd.TargetTTC) && acc_cmd.TargetTTC < cfg_.ttc_hard_guard_s);

  if (!emergency) {
    const double lo = last_a_long_cmd_mps2_ - cfg_.max_jerk_dec_mps3 * dt_s;
    const double hi = last_a_long_cmd_mps2_ + cfg_.max_jerk_acc_mps3 * dt_s;
    a_long_cmd = clampd(a_long_cmd, lo, hi);

    // re-project with fixed alat (keep u_proj.alat) to ensure hard disk constraint
    Acc2 u2{ u_proj.alat, a_long_cmd };
    u2 = ProjectToFeasibleSet(u2, A, alat_strip, w_lat, w_long);

    // also apply measured leftover again
    u2.along = clampd(u2.along, -along_left, +along_left);
    a_long_cmd = u2.along;
    u_proj.alat = u2.alat;
  }

  const bool acc_released_brake =
      !acc_requests_brake &&
      !curve_is_bottleneck &&
      acc_cmd.longitudinal_phase != acc::AccLongitudinalPhase::Braking;
  if (acc_released_brake && a_long_cmd < 0.0) {
    a_long_cmd = 0.0;
  }
  if (acc_coast_request && !curve_is_bottleneck && !acc_requests_brake) {
    a_long_cmd = 0.0;
  }

  last_a_long_cmd_mps2_ = a_long_cmd;

  // 13) convert to speed command
  double v_cmd = std::max(0.0, v + a_long_cmd * dt_s);
  last_speed_cmd_mps_ = v_cmd;

  // 14) steering output from projected alat (signed), convert back to steering-wheel deg
  double steer_out_deg = lka_steer_deg;
  if (v > 0.5) {
    const double kappa_out = u_proj.alat / (v * v);
    const double delta_out = std::atan(kappa_out * cfg_.wheelbase_m); // road wheel rad
    const double delta_out_deg_road = Rad2Deg(delta_out);
    steer_out_deg = delta_out_deg_road * steer_ratio; // back to steering wheel deg
  }
  last_steer_deg_ = steer_out_deg;

  // 15) map to your interface: speed_kmh + brake_0_10
  double brake_out = 0.0;
  double speed_out_kmh = mps2KmH(v_cmd);

  if (a_long_cmd < -0.05) {
    const double decel_need_mps2 = -a_long_cmd;
    const double multiplier = std::max(1e-3f, acc_cfg.brake_multiplier);
    const double full_decel_mps2 = std::max(1e-3f, acc_cfg.brake_full_decel_mps2);
    brake_out = (decel_need_mps2 / full_decel_mps2) * multiplier;
    brake_out = clampd(brake_out, 0.0, 10.0);

    // keep your convention: braking -> speed=0
    speed_out_kmh = 0.0;
  } else if (acc_idle_request && !curve_is_bottleneck) {
    // Idle remains a disabled/invalid command and maps to zero speed.
    speed_out_kmh = 0.0;
  } else if (acc_coast_request && !curve_is_bottleneck) {
    speed_out_kmh = mps2KmH(v);
  }

  out.steer_deg   = static_cast<float>(steer_out_deg);
  out.speed_kmh   = static_cast<float>(std::max(0.0, speed_out_kmh));
  out.brake_0_10  = static_cast<float>(brake_out);

  // debug
  {
    std::ostringstream oss;
    oss << "Supervisor(PROJ) | v=" << mps2KmH(v) << "kmh"
        << " | steer_in=" << lka_steer_deg
        << " steer_out=" << steer_out_deg
        << " | kappa=" << kappa_signed
        << " | alat_nom=" << alat_nom
        << " | alat_meas_abs=" << (std::isfinite(alat_meas_abs) ? alat_meas_abs : -1.0)
        << " | A(mu*g*safe)=" << A
        << " | strip=" << alat_strip
        << " | along_left=" << along_left
        << " | a_long_cmd=" << a_long_cmd
        << " | v_acc_target=" << mps2KmH(v_acc_target) << "kmh"
        << " | v_target_raw=" << mps2KmH(v_target_raw) << "kmh"
        << " | v_target_smoothed=" << mps2KmH(v_target) << "kmh"
        << " | mu_eff=" << mu_eff_ << (in_slip_ ? "(dynamic)" : "(static)")
        << " | v_curve_limit=" << mps2KmH(v_curve_limit) << "kmh"
        << " | KE=" << KE << "J"
        << " | Fc=" << F_c << "N"
        << " | ACC{speed=" << acc_cmd.speed_kmh
        << ", brake=" << acc_cmd.brake_0_10
        << ", TTC=" << acc_cmd.TargetTTC << "}"
        << " | dbgACC=" << acc_dbg
        << " | dbgLKA=" << lka_dbg;
    out.debug = oss.str();
  }

  return out;
}

} // namespace stability
