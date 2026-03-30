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
    std::string run_mode = "video";
    std::string camera_yaml_path = "../Camera-Config/Sensing-3M.yaml";
    std::string icon_path = "../icon";
    float fallback_ego_speed_kmh = 10.0f;
    bool enable_collision_actuation = false;
    bool draw_collision_border = true;
    bool draw_collision_target_box = true;
    bool show_timing_ms = true;
    int wait_key_ms = 30;

    bool enable_keypad_evdev = true;
    std::string keypad_device_path = "/dev/input/event1";

    bool can_tx_master_enable = false;
    bool can_longitudinal_enable = false;
    bool can_steering_enable = false;
    std::string longitudinal_controller = "keypad";

    bool draw_inference_overlay = true;
    bool draw_acc_overlay = true;
    bool draw_lka_overlay = true;
    bool draw_behavior_overlay = true;
    bool draw_collision_overlay = true;
    bool draw_ground_grid_overlay = false;
    bool draw_lane_detect_overlay = false;
    bool draw_status_hud = true;

    float ground_grid_forward_start_m = 1.0f;
    float ground_grid_forward_end_m = 40.0f;
    float ground_grid_lateral_min_m = -8.0f;
    float ground_grid_lateral_max_m = 8.0f;
    float ground_grid_spacing_m = 1.0f;
    float ground_grid_sample_step_m = 0.25f;
    int ground_grid_major_every_n = 5;
    bool ground_grid_draw_labels = true;
};

struct TensorRtRuntimeConfig {
    int topk = 100;
    float score_thres = PROB_THRESHOLD;
    float iou_thres = NMS_THRESHOLD_BBOX;
    int num_labels = NUM_CLASS;
};

struct ModelRuntimeConfig {
    int classify_model_width = Classify_Model_Width;
    int classify_model_height = Classify_Model_Height;
    TensorRtRuntimeConfig tensorrt;
};

struct VehicleBehaviorRuntimeConfig {
    bool enable = true;
    bool use_custom_layout = false;
    std::array<int, 12> custom_layout = {11, 12, 13, 14, 6, 7, 8, 9, 2, 3, 4, 5};
};

struct AblationRuntimeConfig {
    bool enable = true;
    std::string output_path;
    std::string output_dir = "research_logs";
    int flush_every_n = 30;
    int plot_size_px = 1200;
    int plot_margin_px = 80;

    bool virtual_road_enable = true;
    std::string virtual_road_mode = "csv";
    std::string virtual_road_csv_path = "./road_csv/s_curve.csv";
    double virtual_road_length_m = 300.0;
    double virtual_road_step_m = 0.5;
    double virtual_road_lane_width_m = 3.5;
    double virtual_road_arc_radius_m = 120.0;
    double virtual_road_s_amplitude_m = 2.0;
    double virtual_road_s_wavelength_m = 80.0;

    uint64_t virtual_sim_frame_count = 1200;
    double virtual_sim_dt_s = 0.05;
    double virtual_sim_speed_kmh = 30.0;
    double virtual_sim_max_steer_deg = 35.0;
    double virtual_sim_vc_k_cte = 6.0;
    double virtual_sim_vc_k_heading = 0.9;
    double virtual_sim_raw_k_cte = 1.8;
    double virtual_sim_raw_k_heading = 0.2;
    double virtual_sim_raw_steer_bias_deg = 2.5;
    double virtual_sim_raw_steer_osc_amp_deg = 1.0;
    double virtual_sim_raw_steer_osc_period_s = 6.0;
};

struct AdasSystemConfig {
    AppRuntimeConfig app;
    InputViewConfig input;
    GeometryConfig geometry;
    ModelRuntimeConfig model;
    SORTTRACKING::SortTrackingConfig sort;
    sort_kpt::KeypointFilterConfig sort_keypoint;
    acc::AccConfig acc;
    ControlConfig lka;
    stability::StabilityConfig stability;
    collision::CollisionAssistConfig collision;
    VehicleBehaviorRuntimeConfig behavior;
    AblationRuntimeConfig ablation;
};

bool LoadSystemConfig(const std::string& path, AdasSystemConfig& out_config, std::string* out_error = nullptr);
