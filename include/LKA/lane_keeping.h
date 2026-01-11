#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "SortTracking.h"
#include "CameraModel.h"

// =========================
// Public types (API stable)
// =========================

// Controller configuration
struct ControlConfig {
    // --- Vehicle/control base parameters ---
    float wheel_base_m = 2.62f;          //【可調】軸距 L（m）。影響曲率前饋 atan(L·kappa) 與整體轉向幾何；需符合實車/平台。
    float velocity_mps = 5.0f;           //【可調/每幀更新】車速 v（m/s）。Stanley 的橫向誤差項會除以 (v+softening)；速度越高修正越溫和。
    float softening = 0.5f;              //【可調】速度軟化項（m/s）。避免 v→0 時增益發散；調大可抑制低速抖動但會降低低速修正力。

    // --- Stanley dual-model gains ---
    float k_straight = 0.7f;             //【可調】直線模式 Stanley 增益 k。過大易左右擺動；過小會跟線遲鈍/偏移恢復慢。
    float k_curve    = 3.0f;             //【可調】彎道模式 Stanley 增益 k。用於彎道更積極修正；過大可能在彎道震盪或過衝。

    // --- Preview/reference x locations (meters) ---
    float x_ref_straight_m     = 0.30f;  //【可調】直線模式：計算 CTE（橫向誤差）用的前視距離 x_ref（m）。越大越前瞻、越平滑；越小越貼近當前但可能抖動。
    float x_heading_straight_m = 1.50f;  //【可調】直線模式：計算航向誤差用的前視距離 x_heading（m）。越大更穩但反應慢；越小更敏捷但易受雜訊影響。

    float x_ref_curve_m        = 0.30f;  //【可調】彎道模式：CTE 取樣 x_ref（m）。彎道常建議較小以避免「切彎」；但過小會更抖。
    float x_heading_curve_m    = 0.80f;  //【可調】彎道模式：航向誤差取樣 x_heading（m）。越小越貼近曲率、越能跟彎；過小可能放大斜率估計雜訊。

    // --- Feedforward (curvature) ---
    bool  enable_feedforward = true;     //【可調】是否啟用曲率前饋（彎道提早打方向）。通常可降低彎道延遲，但若曲率估計抖動會引入抖動。
    float ff_gain = 1.0f;                //【可調】前饋增益（尺度因子）。>1 會更積極；過大易過度轉向/在彎道過衝。
    float x_curvature_m = 1.00f;         //【保留/目前未使用】原先可用於指定 x 位置取曲率；目前實作使用「平均曲率」mean_kappa。
    float max_ff_deg = 25.0f;            //【可調】前饋角度上限（deg）。防止曲率估計異常時前饋過大。

    // --- Output limits ---
    float max_steer_deg = 30.0f;         //【可調】方向盤/舵角輸出飽和上限（deg）。請依車輛可用轉角與上層安全策略設定。
    float max_steer_rate_deg_s = 200.0f; //【可調】舵角變化率限制（deg/s）。調小更平順但反應慢；調大更靈敏但可能抖動。
    float dt_s = 0.02f;                  //【可調】控制迴圈時間間隔（s）。用於 rate limit；需與實際呼叫週期一致（例如 50 Hz → 0.02s）。

    // --- Keypoint filtering ---
    bool  use_confidence = true;         //【可調】是否使用 kpt.z 作為信心值過濾（需模型輸出 z=confidence）。
    float conf_threshold = 0.5f;         //【可調】最小信心門檻。調高可降低誤點但可能點數不足導致控制保持 last steer。
    float min_x_m = 0.05f;               //【可調】最小前向距離（m）。過小易包含車頭附近雜訊點；過大會失去近距離約束。
    float max_x_m = 30.0f;               //【可調】最大前向距離（m）。過大易受遠端雜訊影響；過小可能在高速/視野短時點數不足。
    float max_abs_y_m = 5.0f;            //【可調】橫向容許範圍 |y|（m）。用於剔除離群點；需大於預期車道線橫向範圍。

    // --- Curve probability metric ---
    int   curvature_samples = 6;         //【可調】曲率取樣點數 M（沿 x 範圍等距取樣）。越大越穩但更慢；過小會使 metric 抖動。
    float metric_w_mean = 1.0f;          //【可調】曲率平均值 |mean_kappa| 權重。調大→更偏向偵測「持續彎道」。
    float metric_w_std  = 0.5f;          //【可調】曲率標準差 std_kappa 權重。調大→更敏感於曲率變化（例如進出彎/點雲抖動）。

    // sigmoid probability
    bool  use_sigmoid_probability = true; //【可調】使用 sigmoid 將 metric 轉為連續 p_curve（0~1），用於直線/彎道控制混合。
    float metric_threshold = 0.08f;       //【可調】sigmoid 門檻。調高→較不易判定為彎道；調低→更容易進入彎道模式。
    float metric_sensitivity = 25.0f;     //【可調】sigmoid 斜率（靈敏度）。越大切換越陡（更像硬切）；越小切換更平滑但可能「半彎半直」。

    // hysteresis probability
    bool  use_hysteresis = false;        //【可調】啟用遲滯（hysteresis）模式：先決定離散彎道/直線，再輸出 0/1（無混合）。可避免頻繁切換。
    float metric_enter_curve = 0.10f;    //【可調】進入彎道門檻（需高於 exit 才有遲滯帶）。
    float metric_exit_curve  = 0.06f;    //【可調】離開彎道門檻。設太高會不易退出；設太低會在彎道中被誤退出。

    // probability low-pass
    bool  enable_prob_lowpass = true;    //【可調】對 p_curve 做一階低通（抑制檢測抖動）。
    float prob_alpha = 0.85f;            //【可調】低通係數 alpha（0~1）。越接近 1 越平滑但反應慢；越小越跟隨即時但易跳動。

    float lane_width_m = 3.76f;         //【可調】車道寬度（m）。僅在「只偵測到單側車道線」時，用來推估中心線偏移量（± lane_width/2）。
};

// Controller state (persistent across frames)
struct ControlState {
    float last_steer_deg = 0.0f;
    float last_steer_rad = 0.0f;

    float p_curve = 0.0f;
    bool  mode_curve = false;

    std::string debug;
};

// =========================
// Public API
// =========================

// Input TrackingBox is assumed to be in vehicle ground frame (meters): x-forward, y-left.
// 重要假設/提醒：
// 1) world_result 的 TrackingBox.kpts 單位為「公尺」，座標定義為 x=前方、y=左方。
// 2) 中心線生成時，若只偵測到單側車道線，內部會假設 lane width = 3.5 m（目前硬編碼於 lk_centerline.cpp）。
float calculate_lane_steering(const TrackingBox& input,
                              const ControlConfig& cfg,
                              ControlState* state);

// Maintain this exact signature to preserve existing call sites, e.g.:
// float steer_deg = lane_steering_step(WorldResult, v, &dbg, Output_frame, Output_frame, &cam);
float lane_steering_step(const std::vector<TrackingBox>& world_result,
                         float velocity_mps,
                         std::string* out_debug = nullptr,
                         cv::Mat input_img = cv::Mat(),
                         cv::Mat output_img = cv::Mat(),
                         const CameraModel* cam = nullptr);
