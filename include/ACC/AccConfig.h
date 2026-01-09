#pragma once
#include <cstdint>

namespace acc {

// ISO 15622 常見採用的控制策略是「固定時距（time headway）」+ 安全限制。
// 參數需要依車輛動力/煞車特性做標定；這裡給可用的工程預設值。
struct AccConfig {
  // 選擇目標物（Lead vehicle）條件
  float lateral_limit_m = 1.0f;      // 對應你提的 abs(x) <= 200cm（約±2m 車道範圍）
  float min_forward_m   = 0.5f;
  float max_forward_m   = 120.0f;

  // 巡航設定
  float cruise_speed_kmh = 60.0f;    // 無前車時的巡航目標

  // ISO風格：固定時距策略
  float time_gap_s       = 1.5f;     // 常用 1.2~2.0s
  float standstill_gap_m = 2.0f;     // 靜止最小距離（stop-and-go）

  // 動態限制（舒適/安全）
  float max_accel_mps2        = 2.0f;  // 最大加速度
  float comfort_decel_mps2    = 2.5f;  // 舒適減速度（IDM會用）
  float max_decel_mps2        = 6.0f;  // 最大減速度（安全層）
  float jerk_limit_mps3       = 2.0f;  // 加加速度限制，避免指令跳動

  // 煞車輸出映射：brake=1 就很大，所以用 brake_full_decel 定義其力度
  float brake_full_decel_mps2 = 3.0f;  // brake=1 約等效 3 m/s^2 減速度（可自行改）
  float brake_multiplier      = 1.0f;  // 你要求可倍數控制煞車

  // 估測與更新
  float default_fps = 30.0f;
  float lead_hysteresis_m = 1.0f;    // 目標切換遲滯，避免跳車

  // 若你有CAN車速可餵進來，精度會明顯提升（但非必要）
  bool  use_external_ego_speed = false;
};

struct AccCommand {
  float speed_kmh   = 0.0f;  // 當作油門端速度命令（brake>0時會被清成0）
  float brake_0_10  = 0.0f;  // 0~10，且 brake=1 就很大
  int   target_id   = -1;    // 除錯用：當前鎖定目標
  float target_forward_m = 0.0f;
  float relative_speed_mps = 0.0f; // lead - ego (由距離微分估測)

  float TargetSpeedKmh = 0.0f;  // 估測的目標車速 (km/h)
  float Targetdistance = 0.0f;  // 估測的目標距離 (m)
  float TargetTTC      = 0.0f;  // TTC (s), closing_speed<=0 時為 inf
};

} // namespace acc

