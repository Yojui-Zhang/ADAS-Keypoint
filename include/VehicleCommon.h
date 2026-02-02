#pragma once

namespace common {

// 1. 車輛物理參數 (Vehicle Physics) - 只有這裡定義一次
struct VehiclePhysicsParams {
    double mass_kg = 1500.0;
    double wheelbase_m = 2.62;      // ACC, LKA, Stability 全部讀這個
    double steering_ratio = 14.5;   // 方向盤轉角 / 路輪轉角
    double center_to_front_m = 1.1; // (可選) 用於碰撞檢測
};

// 2. 動力學極限與舒適性 (Dynamics Constraints) - 統一定義
// 論文中對應：Physical Constraints Set (C)
struct DynamicsConstraints {
    // --- 縱向 (Longitudinal) ---
    double max_accel_mps2 = 2.0;       // [統一] 取代 ACC 的 max_accel
    double comfort_decel_mps2 = 2.5;   // [統一] 取代 ACC 的 comfort_decel
    double max_decel_mps2 = 6.0;       // [統一] 緊急煞車極限
    double jerk_limit_mps3 = 2.0;      // [統一] 加加速度限制

    // --- 橫向 (Lateral) ---
    double max_steer_deg = 450.0;      // 方向盤最大轉角
    double max_steer_rate_deg_s = 400.0; // 方向盤最大轉速
    double max_lat_accel_mps2 = 2.5;   // 舒適側向加速度上限
};

// 3. 系統運行參數 (System Runtime)
struct SystemParams {
    double expected_dt_s = 0.033; // 30Hz
};

} // namespace common