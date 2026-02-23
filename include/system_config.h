#pragma once

#include <array>
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
    std::string camera_yaml_path = "../Camera-Config/Sensing-3M.yaml";
    std::string icon_path = "../icon";
    float fallback_ego_speed_kmh = 10.0f;
    bool enable_collision_actuation = false;
    bool draw_collision_border = true;
    bool draw_collision_target_box = true;
    bool show_timing_ms = true;
    int wait_key_ms = 30;
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
};

bool LoadSystemConfig(const std::string& path, AdasSystemConfig& out_config, std::string* out_error = nullptr);
