#pragma once
#include <cstdint>

namespace stability {

// 監管器參數：把「車輛物理限制」與「舒適性」獨立出來，方便你標定
struct StabilityConfig {
  // --- vehicle ---
  double mass_kg = 1500.0;
  double wheelbase_m = 2.62;

  // steering_ratio: 若你的 steer_deg 是「方向盤角」不是「路輪角」就填入轉向比（例如 14~18）
  // 若你輸出的 steer_deg 已經是路輪角，設 1.0
  double steering_ratio = 1.0;

  // gravity
  double g = 9.81;

  // friction
  double mu_static  = 0.90; // 靜摩擦係數（乾地）
  double mu_dynamic = 0.75; // 動摩擦係數（打滑時）
  double mu_lowpass_alpha = 0.90; // mu 平滑（避免抖動）

  // safety factors（留裕量，避免估測誤差）
  double lat_safety = 0.85;   // 側向裕量
  double total_safety = 0.90; // 摩擦圓總裕量

  // comfort limits
  double lat_accel_comfort_mps2 = 2.5; // 乘坐舒適側向加速度上限
  double long_accel_comfort_mps2 = 1.8; // 舒適加速
  double long_decel_comfort_mps2 = 2.8; // 舒適減速

  // emergency（仍受摩擦圓限制）
  double emergency_decel_cap_mps2 = 6.0; // 允許最大緊急減速（演算法上限）

  // steering envelope
  double steer_high_speed_guard_kmh = 60.0; // 高於此速，強制使用摩擦包絡限制 steer
  double min_speed_for_curvelimit_mps = 1.0;

  // throttle/speed shaping
  double speed_lowpass_alpha = 0.85; // 目標速度平滑
  double max_speed_drop_mps2 = 3.5;  // 每秒最大降速等效（避免突然大煞）
  double max_speed_rise_mps2 = 2.0;  // 每秒最大升速等效

  // slip detect hysteresis（用需求側向加速度逼近 mu*g 判定）
  double slip_enter_ratio = 0.98; // a_lat > mu_s*g*ratio => 進入 slip
  double slip_exit_ratio  = 0.85; // a_lat < mu_s*g*ratio => 離開 slip

  // TTC extra guard（可選：感知誤判時也不要做太激烈）
  double ttc_hard_guard_s = 0.8;  // 小於此 TTC 允許更積極減速（但仍受 mu*g）
};

} // namespace stability

