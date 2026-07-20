#pragma once

#include <opencv2/core.hpp>

struct TargetInfoControlState {
    bool speed_control_active = false;
    bool steering_control_active = false;
    bool brake_control_active = false;
};

void DrawTargetInfo(cv::Mat& img,
                    float current_speed_kmh,
                    float target_speed_kmh,
                    float current_steer_deg,
                    float target_steer_deg,
                    float current_brake_0_10,
                    float target_ttc_s,
                    const TargetInfoControlState& control_state = TargetInfoControlState{});
