#include "StabilitySupervisor.h"
#include "VehiclePhysics.h"
#include <sstream>
#include <cmath>
#include <algorithm>

namespace stability {

static inline double KmH2mps(double kmh) { return kmh / 3.6; }
static inline double mps2KmH(double mps) { return mps * 3.6; }

VehicleControlCommand StabilitySupervisor::Update(double ego_speed_mps,
                                                  double dt_s,
                                                  const acc::AccCommand& acc_cmd,
                                                  double lka_steer_deg,
                                                  const std::string& acc_dbg,
                                                  const std::string& lka_dbg)
{
  VehicleControlCommand out;
  dt_s = std::max(1e-3, dt_s);
  const double v = std::max(0.0, ego_speed_mps);

  // -------------------------
  // 0) 基本防呆
  // -------------------------
  if (!IsFinite(v) || v > 120.0) { // 120 m/s ~ 432 km/h（不合理就鎖死保守）
    out.steer_deg = 0.0f;
    out.speed_kmh = 0.0f;
    out.brake_0_10 = 10.0f;
    out.debug = "invalid ego speed -> full brake safe.";
    return out;
  }
  if (!IsFinite(lka_steer_deg)) lka_steer_deg = last_steer_deg_;

  // -------------------------
  // 1) 把 steer 轉為「路輪角」
  // -------------------------
  const double steer_ratio = std::max(1e-3, cfg_.steering_ratio);
  const double delta_road_deg = lka_steer_deg / steer_ratio;
  const double delta_road_rad = Deg2Rad(delta_road_deg);

  // 曲率/離心
  const double kappa = CurvatureFromSteerRad(delta_road_rad, cfg_.wheelbase_m);
  const double kappa_abs = std::fabs(kappa);
  const double a_lat_demand = LateralAccel(v, kappa_abs);
  const double F_c = CentripetalForce(cfg_.mass_kg, a_lat_demand);
  const double KE = KineticEnergy(cfg_.mass_kg, v);

  // -------------------------
  // 2) 動/靜摩擦切換（用 a_lat 需求逼近 mu*g 判定）
  // -------------------------
  const double mu_s = std::max(0.05, cfg_.mu_static);
  const double mu_k = std::max(0.05, std::min(cfg_.mu_dynamic, mu_s));
  const double g = std::max(1e-3, cfg_.g);

  const double a_lat_s_limit = mu_s * g;

  if (!in_slip_) {
    if (a_lat_demand > a_lat_s_limit * cfg_.slip_enter_ratio) in_slip_ = true;
  } else {
    if (a_lat_demand < a_lat_s_limit * cfg_.slip_exit_ratio) in_slip_ = false;
  }

  const double mu_target = in_slip_ ? mu_k : mu_s;
  const double a_mu = Clamp(cfg_.mu_lowpass_alpha, 0.0, 0.999);
  mu_eff_ = a_mu * mu_eff_ + (1.0 - a_mu) * mu_target;

  // 摩擦圓總上限（含安全裕量）
  const double a_total_max = mu_eff_ * g * cfg_.total_safety;

  // 側向上限再取舒適值
  const double a_lat_allow = std::min(cfg_.lat_accel_comfort_mps2,
                                      mu_eff_ * g * cfg_.lat_safety);

  // -------------------------
  // 3) 由曲率推導「彎道限速」：v <= sqrt(a_lat_allow/|kappa|)
  // -------------------------
  double v_curve_limit = 1e9;
  if (kappa_abs > 1e-6 && v > cfg_.min_speed_for_curvelimit_mps) {
    v_curve_limit = std::sqrt(std::max(0.0, a_lat_allow / kappa_abs));
  }

  // -------------------------
  // 4) 摩擦圓：a_lat^2 + a_long^2 <= a_total_max^2
  //    => a_long_allow = sqrt(a_total_max^2 - a_lat^2)
  // -------------------------
  const double a_lat_used_for_circle = std::min(a_lat_demand, a_total_max);
  double a_long_allow = std::sqrt(std::max(0.0, a_total_max * a_total_max
                                                - a_lat_used_for_circle * a_lat_used_for_circle));

  // 舒適限制（加/減速不同）
  const double a_long_accel_allow = std::min(a_long_allow, cfg_.long_accel_comfort_mps2);
  double a_long_decel_allow = std::min(a_long_allow, cfg_.long_decel_comfort_mps2);

  // 可選：TTC 很小時允許更大減速（但仍不能超過摩擦圓）
  if (std::isfinite(acc_cmd.TargetTTC) && acc_cmd.TargetTTC < cfg_.ttc_hard_guard_s) {
    a_long_decel_allow = std::min(a_long_allow, std::min(cfg_.emergency_decel_cap_mps2, a_total_max));
  }

  // -------------------------
  // 5) 解析 ACC 想要的縱向需求（speed/brake）
  // -------------------------
  const acc::AccConfig acc_cfg = acc::ACC_GetConfig();

  // ACC speed_cmd（若 ACC 在煞車時 speed_kmh 會清 0；你原本就是這樣） :contentReference[oaicite:4]{index=4}
  double v_acc_target = KmH2mps(std::max(0.0f, acc_cmd.speed_kmh));

  // 反推 ACC brake 對應的減速度需求（與 AccController 裡的 mapping 互逆） :contentReference[oaicite:5]{index=5}
  const double brake = Clamp(acc_cmd.brake_0_10, 0.0, 10.0);
  double a_brake_need = 0.0;
  if (brake > 1e-3) {
    const double mult = std::max(1e-3f, acc_cfg.brake_multiplier);
    const double full = std::max(1e-3f, acc_cfg.brake_full_decel_mps2);
    a_brake_need = (brake / mult) * full; // m/s^2
  }

  // 如果 ACC 已要求煞車，v_acc_target 可能是 0；我們依「減速度需求」比較合理
  // 若未煞車，則用 v_acc_target 與當前 v 算 accel_need
  double a_long_need = 0.0;
  if (a_brake_need > 1e-3) {
    a_long_need = -a_brake_need;
  } else {
    a_long_need = (v_acc_target - v) / dt_s;
  }

  // -------------------------
  // 6) Supervisor 對 speed：取 min(ACC目標, 彎道限速)，再做平滑與加速度限制
  // -------------------------
  double v_target = std::min(v_acc_target, v_curve_limit);

  // 速度平滑（避免因瞬間曲率抖動造成油門忽快忽慢）
  const double sp_a = Clamp(cfg_.speed_lowpass_alpha, 0.0, 0.999);
  if (last_speed_cmd_mps_ <= 1e-6) last_speed_cmd_mps_ = v_target;
  v_target = sp_a * last_speed_cmd_mps_ + (1.0 - sp_a) * v_target;

  // 用「等效加速度」限制每秒升/降速幅度
  const double dv_up   = cfg_.max_speed_rise_mps2 * dt_s;
  const double dv_down = cfg_.max_speed_drop_mps2 * dt_s;
  v_target = Clamp(v_target, last_speed_cmd_mps_ - dv_down, last_speed_cmd_mps_ + dv_up);

  // -------------------------
  // 7) 摩擦圓約束下，限制縱向加/減速度
  // -------------------------
  double a_long_cmd = (v_target - v) / dt_s;

  // 若 ACC 明確要求更大減速（a_long_need更負），Supervisor 仍會受摩擦圓限制，避免彎中鎖死打滑
  // 但在直線/低側向時，仍會盡量跟上 ACC 的減速需求
  if (a_long_need < a_long_cmd) a_long_cmd = a_long_need;

  // clamp by friction circle + comfort
  if (a_long_cmd >= 0.0) {
    a_long_cmd = std::min(a_long_cmd, a_long_accel_allow);
  } else {
    a_long_cmd = std::max(a_long_cmd, -a_long_decel_allow);
  }

  // 由 a_long_cmd 回推最終 v_target
  v_target = std::max(0.0, v + a_long_cmd * dt_s);
  last_speed_cmd_mps_ = v_target;

  // -------------------------
  // 8) 轉向安全包絡：高速時限制 steer，使 a_lat 不超過 a_lat_allow
  // -------------------------
  double steer_out_deg = lka_steer_deg;

  const double v_kmh = mps2KmH(v);
  if (v_kmh >= cfg_.steer_high_speed_guard_kmh && kappa_abs > 1e-8) {
    // a_lat_allow = v^2 * |kappa|, kappa = tan(delta)/L
    // => |tan(delta)| <= a_lat_allow*L / v^2
    const double lim = (a_lat_allow * cfg_.wheelbase_m) / std::max(1e-3, v * v);
    const double delta_lim_rad = std::atan(std::max(0.0, lim));
    const double delta_lim_deg = Rad2Deg(delta_lim_rad);

    // 限制的是「路輪角」，再轉回你輸入的 steer_deg（乘 steering_ratio）
    const double steer_lim_deg_cmd = delta_lim_deg * steer_ratio;

    steer_out_deg = Clamp(steer_out_deg, -steer_lim_deg_cmd, +steer_lim_deg_cmd);
  }

  last_steer_deg_ = steer_out_deg;

  // -------------------------
  // 9) 轉成你車上介面：speed_kmh + brake_0_10
  // -------------------------
  double brake_out = 0.0;
  double speed_out_kmh = mps2KmH(v_target);

  if (a_long_cmd < -0.05) {
    const double decel_need = -a_long_cmd;
    const double mult = std::max(1e-3f, acc_cfg.brake_multiplier);
    const double full = std::max(1e-3f, acc_cfg.brake_full_decel_mps2);
    brake_out = (decel_need / full) * mult;
    brake_out = Clamp(brake_out, 0.0, 10.0);

    // 你的 ACC 介面習慣「煞車時 speed=0」就維持一致（避免上層解析衝突） :contentReference[oaicite:6]{index=6}
    speed_out_kmh = 0.0;
  }

  out.steer_deg = static_cast<float>(steer_out_deg);
  out.speed_kmh = static_cast<float>(std::max(0.0, speed_out_kmh));
  out.brake_0_10 = static_cast<float>(brake_out);

  // debug（含離心力/動能/摩擦狀態）
  {
    std::ostringstream oss;
    oss << "Supervisor | v=" << v_kmh << "kmh"
        << " | steer_in=" << lka_steer_deg << "deg"
        << " steer_out=" << steer_out_deg << "deg"
        << " | kappa=" << kappa
        << " | a_lat=" << a_lat_demand << " m/s2"
        << " (allow=" << a_lat_allow << ")"
        << " | a_long_cmd=" << a_long_cmd
        << " (acc_allow=" << a_long_accel_allow
        << " dec_allow=" << a_long_decel_allow << ")"
        << " | mu_eff=" << mu_eff_ << (in_slip_ ? "(dynamic)" : "(static)")
        << " | v_curve_limit=" << mps2KmH(v_curve_limit) << "kmh"
        << " | KE=" << KE << "J"
        << " | Fc=" << F_c << "N"
        << " | ACC{speed=" << acc_cmd.speed_kmh << ", brake=" << acc_cmd.brake_0_10
        << ", TTC=" << acc_cmd.TargetTTC << "}"
        << " | dbgACC=" << acc_dbg
        << " | dbgLKA=" << lka_dbg;

    out.debug = oss.str();
  }

  return out;
}

} // namespace stability

