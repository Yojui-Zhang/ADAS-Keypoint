#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "config.h"
#include "input-view.h"
#include "GeometryFunction.h"
#include "SortTracking.h"
#include "KeypointFilterSwitch.h"
#include "AccConfig.h"
#include "lane_keeping.h"
#include "StabilityConfig.h"
#include "CollisionAssistApi.h"

struct AppRuntimeConfig {
    // 執行模式：video / virtual_road / real_car
    std::string run_mode = "video";
    // 相機內外參檔案路徑（Geometry 轉換依賴）
    std::string camera_yaml_path = "../Camera-Config/Sensing-3M.yaml";
    // 分類圖示資料夾（交通號誌/燈號繪圖）
    std::string icon_path = "../icon";
    // 無 CANBus 時的自車速度回退值（km/h）
    float fallback_ego_speed_kmh = 10.0f;
    // 是否允許碰撞模組直接改寫 speed/steer/brake
    bool enable_collision_actuation = false;
    // 是否繪製碰撞警示紅框（全螢幕邊框）
    bool draw_collision_border = true;
    // 是否標示碰撞威脅目標框
    bool draw_collision_target_box = true;
    // 是否輸出每幀耗時（ms）
    bool show_timing_ms = true;
    // `cv::waitKey()` 延遲（ms）
    int wait_key_ms = 30;
};

struct TensorRtRuntimeConfig {
    // TensorRT 後處理：每幀保留候選框上限
    int topk = 100;
    // TensorRT 後處理：分數門檻
    float score_thres = PROB_THRESHOLD;
    // TensorRT 後處理：NMS IoU 門檻
    float iou_thres = NMS_THRESHOLD_BBOX;
    // TensorRT 模型類別數
    int num_labels = NUM_CLASS;
};

struct ModelRuntimeConfig {
    // 分類模型輸入寬度（像素）
    int classify_model_width = Classify_Model_Width;
    // 分類模型輸入高度（像素）
    int classify_model_height = Classify_Model_Height;
    // TensorRT 專屬推論後處理參數
    TensorRtRuntimeConfig tensorrt;
};

struct VehicleBehaviorRuntimeConfig {
    // 是否啟用 Vehicle Skeleton + Heading 模組
    bool enable = true;
    // 是否使用自訂 keypoint 佈局索引
    bool use_custom_layout = false;
    // 自訂佈局索引（12 點：top/mid/bot × left/right）
    std::array<int, 12> custom_layout = {11, 12, 13, 14, 6, 7, 8, 9, 2, 3, 4, 5};
};

struct AblationRuntimeConfig {
    // 演算法比對 logger 總開關
    bool enable = true;
    // 輸出 CSV 完整路徑（留空則使用 output_dir + 時間戳）
    std::string output_path;
    // 輸出資料夾
    std::string output_dir = "research_logs";
    // 每 N 幀 flush 一次檔案，降低資料遺失風險
    int flush_every_n = 30;
    // 路徑圖輸出尺寸（像素）
    int plot_size_px = 1200;
    // 路徑圖留白邊界（像素）
    int plot_margin_px = 80;

    // 是否啟用 reference road 比對（CTE/heading/lane departure）
    bool virtual_road_enable = true;
    // 參考道路模式：straight / arc / s_curve / csv
    std::string virtual_road_mode = "csv";
    // mode=csv 時讀取的道路檔案（每列 x,y）
    std::string virtual_road_csv_path = "./road_csv/s_curve.csv";
    // 內建道路（非 csv 模式）總長（m）
    double virtual_road_length_m = 300.0;
    // 內建道路（非 csv 模式）取樣間距（m）
    double virtual_road_step_m = 0.5;
    // 車道寬（m），用於偏離車道判定
    double virtual_road_lane_width_m = 3.5;
    // 圓弧半徑（m），arc 模式使用，正值左彎負值右彎
    double virtual_road_arc_radius_m = 120.0;
    // S 彎振幅（m），s_curve 模式使用
    double virtual_road_s_amplitude_m = 2.0;
    // S 彎波長（m），s_curve 模式使用
    double virtual_road_s_wavelength_m = 80.0;

    // 虛擬道路閉迴路模擬（run_mode=virtual_road）
    uint64_t virtual_sim_frame_count = 1200;         // 模擬幀數
    double virtual_sim_dt_s = 0.05;                  // 模擬 dt（s）
    double virtual_sim_speed_kmh = 30.0;             // 模擬固定速度（km/h）
    double virtual_sim_max_steer_deg = 35.0;         // 模擬最大轉角（deg）
    double virtual_sim_vc_k_cte = 6.0;               // VC on：CTE 控制增益
    double virtual_sim_vc_k_heading = 0.9;           // VC on：heading 控制增益
    double virtual_sim_raw_k_cte = 1.8;              // VC off：CTE 控制增益
    double virtual_sim_raw_k_heading = 0.2;          // VC off：heading 控制增益
    double virtual_sim_raw_steer_bias_deg = 2.5;     // VC off：固定轉向偏置（deg）
    double virtual_sim_raw_steer_osc_amp_deg = 1.0;  // VC off：轉向擾動振幅（deg）
    double virtual_sim_raw_steer_osc_period_s = 6.0; // VC off：轉向擾動週期（s）
};

struct AdasSystemConfig {
    // 全域應用行為（時間、UI、碰撞介入等）
    AppRuntimeConfig app;
    // 影像來源與顯示設定
    InputViewConfig input;
    // 像素到世界座標轉換設定
    GeometryConfig geometry;
    // 模型與引擎設定
    ModelRuntimeConfig model;

    // SORT 追蹤參數
    SORTTRACKING::SortTrackingConfig sort;
    // SORT keypoint 濾波參數（EMA/KF）
    sort_kpt::KeypointFilterConfig sort_keypoint;

    // ACC 縱向控制參數
    acc::AccConfig acc;
    // LKA 橫向控制參數
    ControlConfig lka;
    // Stability 監管器參數（摩擦圓/舒適/速率限制）
    stability::StabilityConfig stability;
    // Collision 風險評估與介入參數
    collision::CollisionAssistConfig collision;
    // Vehicle behavior（骨架/朝向）設定
    VehicleBehaviorRuntimeConfig behavior;
    // 模擬/論文分析用 ablation 記錄與虛擬道路設定
    AblationRuntimeConfig ablation;
};

bool LoadSystemConfig(const std::string& path, AdasSystemConfig& out_config, std::string* out_error = nullptr);
