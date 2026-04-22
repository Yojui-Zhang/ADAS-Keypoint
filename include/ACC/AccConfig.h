#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace acc {

enum class AccTrackedObjectState : std::uint8_t {
  Remaining = 0,
  Candidate = 1,
  Lead = 2,
  FollowingLead = 3,
};

enum class AccLongitudinalPhase : std::uint8_t {
  MaxHold = 0,
  Accelerating = 1,
  Idle = 2,
  Braking = 3,
};

inline const char* AccTrackedObjectStateName(AccTrackedObjectState state) {
  switch (state) {
    case AccTrackedObjectState::Candidate: return "candidate";
    case AccTrackedObjectState::Lead: return "lead";
    case AccTrackedObjectState::FollowingLead: return "following_lead";
    default: return "remaining";
  }
}

inline int AccTrackedObjectStateCode(AccTrackedObjectState state) {
  return static_cast<int>(state);
}

inline const char* AccLongitudinalPhaseName(AccLongitudinalPhase phase) {
  switch (phase) {
    case AccLongitudinalPhase::MaxHold: return "max_hold";
    case AccLongitudinalPhase::Accelerating: return "accelerating";
    case AccLongitudinalPhase::Braking: return "braking";
    default: return "idle";
  }
}

inline int AccLongitudinalPhaseCode(AccLongitudinalPhase phase) {
  return static_cast<int>(phase);
}

// ISO 15622 常見採用的控制策略是「固定時距（time headway）」+ 安全限制。
// 參數需要依車輛動力/煞車特性做標定；這裡給可用的工程預設值。
struct AccConfig {
  // ==========================================
  // 1. 感知與目標選擇 (Target Selection)
  // ==========================================
  
  // 車道橫向限制 (公尺)
  // 影響：決定 ACC 會抓多寬範圍內的車。
  // 若當前幀可建立本車道中心線，這個值代表「相對於中心線」的容許半寬；
  // 若當前幀抓不到車道，則退回成「相對於自車 y=0」的容許半寬。
  // 設定建議：一般車道寬約 3.5m~3.75m，設 1.0f~1.5f 代表只抓正前方本車道內的車。
  // 若設太大 (如 3.0f)，過彎或鄰車道車輛靠近時容易誤煞車。
  float lateral_limit_m = 1.0f;      

  // 最小偵測距離 (公尺)
  // 影響：太近的目標 (如引擎蓋前的雜訊) 會被過濾掉。
  float min_forward_m   = 0.5f;

  // 最大有效距離 (公尺)
  // 影響：ACC 考慮的最遠距離。超過此距離的車輛只追蹤但不進行加減速反應。
  float max_forward_m   = 40.0f;

  // 目標切換遲滯 (公尺) - 用於 LeadSelector
  // 影響：防止鎖定目標在前車與旁車之間快速跳動。
  // 若新目標比舊目標只近了一點點 (小於此值)，則維持鎖定舊目標。
  float lead_hysteresis_m = 1.0f;    


  // ==========================================
  // 2. 縱向控制策略 (Longitudinal Control)
  // ==========================================

  // 定速巡航車速 (km/h)
  // 影響：前方無車時，車輛會加速到的最高速度。
  float cruise_speed_kmh = 30.0f;    

  // 跟車時距 (秒) - Time Gap
  // 影響：決定跟車距離的鬆緊。公式：距離 = 靜止間距 + 車速 * time_gap。
  // 值越大，跟車距離越遠 (安全)；值越小，跟得越緊 (容易被插隊)。
  // 建議範圍：1.2s (緊湊) ~ 2.0s (舒適)。例如時速 60km/h (16.6m/s) * 1.5s = 約 25m 車距。
  float time_gap_s       = 1.5f;     

  // 靜止停等距離 (公尺) - Standstill Gap
  // 影響：當前車完全停下時，本車預計停在離前車多遠的地方 (Stop-and-Go)。
  // 建議：2.0m ~ 3.0m。
  float standstill_gap_m = 5.0f;     


  // ==========================================
  // 3. 動力學限制 (Dynamics Limits)
  // ==========================================

  // 最大加速度 (m/s^2)
  // 影響：油門全開時的加速感上限。
  // 建議：一般乘用車舒適值約 1.5~2.0。設太高乘客會覺得猛衝。
  float max_accel_mps2        = 2.0f;  

  // 舒適減速度 (m/s^2)
  // 影響：一般跟車調節速度時的煞車力道基準。
  // 數值越大，ACC 在看到前車減速時會更晚、更急地煞車；數值越小，會更早、更平緩地減速。
  float comfort_decel_mps2    = 2.5f;  

  // 最大減速度 (m/s^2) - 安全底線
  // 影響：緊急情況 (如前車急煞) 允許輸出的最大煞車力道。
  // 注意：這只是演算法的限制，實際煞車力還受限於 brake_full_decel_mps2 的定義。
  float max_decel_mps2        = 6.0f;  

  // 加加速度限制 (m/s^3) - Jerk Limit
  // 影響：加速度變化的快慢，即「頓挫感」的抑制。
  // 值越小，加減速切換越柔順 (像老司機)；值越大，反應越快但越頓。
  // 建議：1.0 ~ 2.5 之間。
  float jerk_limit_mps3       = 2.0f;  


  // ==========================================
  // 4. 執行器映射 (Actuation Mapping)
  // ==========================================

  // 煞車滿載定義 (m/s^2)
  // 影響：將演算法計算出的減速度 (m/s^2) 轉換為 brake (0~10) 訊號。
  // 定義：當輸出 brake = 10 (或歸一化的 1.0) 時，對應物理上多少減速度。
  // 若發現 ACC 煞車常常煞不住，可能需要將此值調「小」(讓同樣的減速需求對應更大的 brake 值) 或調整 multiplier。
  float brake_full_decel_mps2 = 2.0f;  

  // 煞車力道倍率
  // 影響：直接線性放大輸出的 brake 數值。
  // 若車輛煞車較軟，可將此值設為 1.2 或 1.5 來增強輸出。
  float brake_multiplier      = 1.0f;  

  // ==========================================
  // 4.1 怠速滑行 / 提前減速 (Coast & Early Brake)
  // ==========================================

  // 油門死區：加速度需求低於此值時輸出 speed=0, brake=0，代表放油門滑行。
  float throttle_accel_deadband_mps2 = 0.15f;

  // 煞車死區：減速度需求超過此值才轉成 brake_0_10。
  float brake_accel_deadband_mps2 = 0.20f;

  // 當前車進入「期望距離 + 這段額外距離」時，先禁止繼續加油門。
  float coast_gap_margin_m = 3.0f;

  // 額外提前滑行時距。速度越高，越早進入放油門滑行。
  float coast_time_gap_margin_s = 0.8f;

  // 當前車距離低於「期望距離 + 這段 margin」時，提前建立煞車需求。
  float brake_gap_margin_m = 0.0f;

  // 距離不足時，每少 1m 對應多少減速度需求。
  float gap_error_decel_gain_mps2_per_m = 0.35f;

  // 進入提前煞車區時的最小減速度需求，避免只輸出太小的煞車值。
  float min_brake_decel_mps2 = 0.35f;

  // TTC 低於此值開始提前煞車；低於 hard guard 時直接允許最大減速度。
  float ttc_soft_brake_s = 3.0f;
  float ttc_hard_brake_s = 1.5f;

  // 高速跟車放寬：高速下若前車沒有明顯接近，先滑行，不因 time_gap 不足直接煞車。
  bool high_speed_relax_enable = true;
  float high_speed_relax_min_kmh = 40.0f;
  float high_speed_brake_time_gap_s = 0.55f;
  float high_speed_brake_gap_margin_m = 2.0f;
  float high_speed_brake_closing_mps = 1.0f;


  // ==========================================
  // 5. 系統與更新 (System)
  // ==========================================

  // 預設幀率 (FPS)
  // 影響：當無法透過 Frame ID 計算 dt 時，使用此數值估算時間差。
  float default_fps = 30.0f;

  // 使用外部車速訊號
  // True: 使用 ACC_SetEgoSpeedKmh 傳入的 CAN 車速 (推薦，控制較穩)。
  // False: 控制器內部自行積分加速度來估算車速 (會有累積誤差，僅供測試用)。
  bool  use_external_ego_speed = false;
};

struct AccCommand {
  float speed_kmh   = 0.0f;  // 當作油門端速度命令（brake>0時會被清成0）
  float brake_0_10  = 0.0f;  // 0~10，且 brake=1 就很大 (視實車介面定義而定)
  int   target_id   = -1;    // 除錯用：當前鎖定目標 ID
  float target_forward_m = 0.0f; // 濾波後的目標距離
  float relative_speed_mps = 0.0f; // lead - ego (負值代表接近中)

  float TargetSpeedKmh = 0.0f;  // 估測的前車車速 (km/h)
  float Targetdistance = 0.0f;  // 估測的目標距離 (m) (同 target_forward_m)
  float TargetTTC      = 0.0f;  // TTC (s), closing_speed<=0 時為 inf

  float TargetScore   = 0.0f;  // detection/track confidence
  float TargetDistStd = 0.0f;  // sqrt(P00)
  float RelSpeedStd   = 0.0f;  // sqrt(P11)
  float TargetTTCStd  = 0.0f;  // propagated

  bool has_lead = false;
  bool lead_following_active = false; // 當前 lead 已經實際影響縱向控制
  float ego_speed_kmh = 0.0f;
  float cruise_speed_kmh = 0.0f;
  float accel_cmd_mps2 = 0.0f;
  float free_accel_nom_mps2 = 0.0f;
  float free_accel_limited_mps2 = 0.0f;
  float lead_lateral_m = 0.0f;
  AccLongitudinalPhase longitudinal_phase = AccLongitudinalPhase::Idle;

  std::vector<int> candidate_ids;     // 本幀通過 ACC 候選篩選的目標 ID
};

inline bool AccHasCandidateId(const AccCommand& cmd, int id) {
  return std::find(cmd.candidate_ids.begin(), cmd.candidate_ids.end(), id) != cmd.candidate_ids.end();
}

inline AccTrackedObjectState ClassifyAccTrackedObjectState(const AccCommand& cmd, int id) {
  if (id < 0) return AccTrackedObjectState::Remaining;
  if (id == cmd.target_id) {
    return cmd.lead_following_active ? AccTrackedObjectState::FollowingLead
                                     : AccTrackedObjectState::Lead;
  }
  if (AccHasCandidateId(cmd, id)) return AccTrackedObjectState::Candidate;
  return AccTrackedObjectState::Remaining;
}

} // namespace acc
