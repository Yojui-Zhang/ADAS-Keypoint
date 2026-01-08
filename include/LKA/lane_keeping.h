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
    float wheel_base_m = 0.30f;          // wheelbase L (m)
    float velocity_mps = 5.0f;           // vehicle speed v (m/s)
    float softening = 0.5f;              // avoid v->0 divergence: atan(k*e/(v+softening))

    // --- Stanley dual-model gains ---
    float k_straight = 0.7f;             // straight mode: lower gain
    float k_curve    = 3.0f;             // curve mode: higher gain

    // --- Preview/reference x locations (meters) ---
    float x_ref_straight_m     = 0.30f;
    float x_heading_straight_m = 1.50f;

    float x_ref_curve_m        = 0.30f;
    float x_heading_curve_m    = 0.80f;

    // --- Feedforward (curvature) ---
    bool  enable_feedforward = true;
    float ff_gain = 1.0f;
    float x_curvature_m = 1.00f;
    float max_ff_deg = 25.0f;

    // --- Output limits ---
    float max_steer_deg = 30.0f;
    float max_steer_rate_deg_s = 200.0f;
    float dt_s = 0.02f;

    // --- Keypoint filtering ---
    bool  use_confidence = true;
    float conf_threshold = 0.5f;
    float min_x_m = 0.05f;
    float max_x_m = 30.0f;
    float max_abs_y_m = 5.0f;

    // --- Curve probability metric ---
    int   curvature_samples = 6;
    float metric_w_mean = 1.0f;
    float metric_w_std  = 0.5f;

    // sigmoid probability
    bool  use_sigmoid_probability = true;
    float metric_threshold = 0.08f;
    float metric_sensitivity = 25.0f;

    // hysteresis probability
    bool  use_hysteresis = false;
    float metric_enter_curve = 0.10f;
    float metric_exit_curve  = 0.06f;

    // probability low-pass
    bool  enable_prob_lowpass = true;
    float prob_alpha = 0.85f;
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
