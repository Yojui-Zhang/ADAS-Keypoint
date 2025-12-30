#pragma once
#ifndef KEYPOINT_KF_H_
#define KEYPOINT_KF_H_

#include <map>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace kpt_kf {

// ------------------------------------------------------------
// A lightweight 2D constant-velocity Kalman filter for keypoints.
// State: [x, y, vx, vy]^T
// Meas : [x, y]^T
// ------------------------------------------------------------
class Kpt2DKF {
public:
    Kpt2DKF();

    void reset();
    bool isInitialized() const { return inited_; }

    // Initialize filter with a position measurement
    void init(float x, float y,
              float init_pos_var, float init_vel_var,
              float process_var);

    // Predict step. Returns predicted position.
    cv::Point2f predict();

    // Correct step with measurement. meas_var is the measurement variance (px^2).
    cv::Point2f correct(float mx, float my, float meas_var);

    cv::Point2f last() const { return last_xy_; }

private:
    void setupModel_();

    cv::KalmanFilter kf_;
    cv::Mat meas_;
    bool inited_;
    float process_var_;
    cv::Point2f last_xy_;
};

// ------------------------------------------------------------
// Track-wise keypoint filtering manager.
// Input keypoint format: cv::Point3f(x, y, v)
// - v is confidence/visibility in [0,1] (or [0,100] if you use that; then adjust).
// Output keypoint format: cv::Point3f(x_filtered, y_filtered, v_out)
// - If this frame has no valid measurement, v_out will be 0.
// ------------------------------------------------------------
class KeypointKFFilter {
public:
    struct Params {
        float conf_thr;       // below this, treat as low confidence / missing
        float process_var;    // process noise variance (px^2) for all state dims
        float meas_var_base;  // base measurement variance (px^2); actual = base / max(v, eps)
        float gate_dist_px;   // reject measurement if |z - pred| > gate_dist_px (px)

        float init_pos_var;   // initial position variance (px^2)
        float init_vel_var;   // initial velocity variance

        Params(float conf_thr_ = 0.2f,
               float process_var_ = 1.0f,
               float meas_var_base_ = 25.0f,
               float gate_dist_px_ = 80.0f,
               float init_pos_var_ = 100.0f,
               float init_vel_var_ = 1000.0f)
            : conf_thr(conf_thr_),
              process_var(process_var_),
              meas_var_base(meas_var_base_),
              gate_dist_px(gate_dist_px_),
              init_pos_var(init_pos_var_),
              init_vel_var(init_vel_var_) {}
    };

    KeypointKFFilter();
    explicit KeypointKFFilter(const Params& params);

    // Update a track's keypoints.
    // - kpts == nullptr means "no measurements for this track in this frame" (predict-only).
    void Update(int track_id, const std::vector<cv::Point3f>* kpts);

    // Get last output keypoints (filtered) for a track_id.
    bool GetOutput(int track_id, std::vector<cv::Point3f>& out) const;

    // Remove a track's KF states.
    void Erase(int track_id);

    // Remove all states.
    void Clear();

private:
    struct TrackState {
        std::vector<Kpt2DKF> kfs;
        std::vector<cv::Point3f> last_out;
        bool initialized;

        TrackState() : initialized(false) {}
    };

    Params params_;
    std::map<int, TrackState> tracks_;
};

} // namespace kpt_kf

#endif // KEYPOINT_KF_H_
