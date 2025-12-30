#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include "SortTracking.h"

// 你的輸入結構（若你專案已有定義，可改成 include 你的 header）
// typedef struct TrackingBox {
//     int frame;
//     int id;
//     int class_id;
//     float score;
//     cv::Rect box;
//     std::vector<cv::Point3f> kpts; // (x, y, conf) 或 (x,y,z). 本版預設 z=conf（可關閉）
// } TrackingBox;

// 控制器設定
struct ControlConfig {
    // --- 車輛/控制基本參數 ---
    float wheel_base_m = 0.30f;          // 軸距 L (m)
    float velocity_mps = 5.0f;           // 車速 v (m/s)
    float softening = 0.5f;              // 避免 v->0 發散：atan(k*e/(v+softening))

    // --- Stanley 兩模型增益 ---
    float k_straight = 0.7f;             // 直線模式：低增益
    float k_curve    = 3.0f;             // 彎道模式：高增益

    // --- 兩模型的取樣位置（Preview / Reference X）---
    // 直線模型看更遠：抗抖
    float x_ref_straight_m     = 0.30f;  // CTE 評估位置（通常用前軸或其附近）
    float x_heading_straight_m = 1.50f;  // heading 評估位置（遠一點）

    // 彎道模型看較近：緊咬
    float x_ref_curve_m        = 0.30f;
    float x_heading_curve_m    = 0.80f;

    // --- Feedforward (D)：曲率前饋 ---
    bool  enable_feedforward = true;
    float ff_gain = 1.0f;                // 前饋倍率（通常 0.7~1.3）
    float x_curvature_m = 1.00f;         // 用哪個 x 位置估曲率（或路徑中段）
    float max_ff_deg = 25.0f;            // 前饋最大角（避免錯誤點位造成暴衝）

    // --- 輸出限制 ---
    float max_steer_deg = 30.0f;         // 總轉角限制
    float max_steer_rate_deg_s = 200.0f; // 轉角變化率限制（deg/s），D項：抑制抖動/致動器限制
    float dt_s = 0.02f;                  // 控制迴圈週期（秒）

    // --- Keypoints 使用設定 ---
    bool  use_confidence = true;
    float conf_threshold = 0.5f;         // kp.z < threshold 則略過（若 use_confidence=true）
    float min_x_m = 0.05f;               // 過濾極近點（避免相機投影不穩或噪聲）
    float max_x_m = 30.0f;               // 過濾過遠點
    float max_abs_y_m = 5.0f;            // 過濾橫向離譜點

    // --- (C)：模式權重（彎道機率）估測 ---
    // 使用「曲率強度 + 曲率變異」混合指標（metric）
    int   curvature_samples = 6;         // 取幾個 x 位置估 κ
    float metric_w_mean = 1.0f;          // |mean(kappa)| 權重（偏向彎道強度）
    float metric_w_std  = 0.5f;          // std(kappa) 權重（偏向變異度）

    // sigmoid 模式（連續機率）
    bool  use_sigmoid_probability = true;
    float metric_threshold = 0.08f;      // metric 門檻（需依你的 κ 尺度調）
    float metric_sensitivity = 25.0f;    // sigmoid 斜率

    // hysteresis 模式（兩段門檻，較不抖）
    bool  use_hysteresis = false;
    float metric_enter_curve = 0.10f;    // 進彎門檻（高）
    float metric_exit_curve  = 0.06f;    // 出彎門檻（低）

    // 低通濾波（強烈建議開）
    bool  enable_prob_lowpass = true;
    float prob_alpha = 0.85f;            // p = alpha*p_prev + (1-alpha)*p_raw
};

// 控制器狀態（跨幀記憶）
struct ControlState {
    float last_steer_deg = 0.0f;
    float last_steer_rad = 0.0f;

    float p_curve = 0.0f;        // 濾波後彎道機率
    bool  mode_curve = false;    // hysteresis 用的離散模式（可選）

    std::string debug;           // 方便你 log/印出
};

// 對外 API：輸入 lane tracking result，輸出 steering angle（degrees）
// 注意：本版本假設 input.kpts 已是車體地面座標（m），x前 y左。
float calculate_lane_steering(const TrackingBox& input,
                              const ControlConfig& cfg,
                              ControlState* state);
