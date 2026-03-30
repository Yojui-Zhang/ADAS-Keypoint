#pragma once

#include <opencv2/core.hpp>

class CameraModel;

struct WorldGridOverlayConfig {
    bool enabled = false;
    float forward_start_m = 1.0f;
    float forward_end_m = 40.0f;
    float lateral_min_m = -8.0f;
    float lateral_max_m = 8.0f;
    float spacing_m = 1.0f;
    float sample_step_m = 0.25f;
    int major_every_n = 5;
    bool draw_labels = true;
};

void DrawWorldGridOverlay(cv::Mat& image,
                          const CameraModel& cam,
                          const WorldGridOverlayConfig& cfg);
