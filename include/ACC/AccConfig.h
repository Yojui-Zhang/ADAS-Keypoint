#pragma once
#include <cstdint>

namespace acc {

// ISO 15622 常見採用的控制策略是「固定時距（time headway）」+ 安全限制。
// 參數需要依車輛動力/煞車特性做標定；這裡給可用的工程預設值。
struct AccConfig {
  // ==========================================
  // 1. 感知與目標選擇 (Target Selection)
  // ==========================================
  
  // 車道橫向限制 (公尺)
  // 影響：決定 ACC 會抓多寬範圍內的車。
  // 設定建議：一般車道寬約 3.5m~3.75m，設 1.0f~1.5f 代表只抓正前方左右偏移 1~1.5m 的車 (即本車道)。
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
};

} // namespace acc