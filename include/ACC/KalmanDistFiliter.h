#pragma once

#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace acc {

class AccKalmanAdapter {
public:
  AccKalmanAdapter() = default;

  // tuning
  void SetTuning(float meas_std_m,
                 float accel_std_mps2,
                 float gate_nis_soft = 9.0f,
                 float gate_nis_hard = 25.0f) {
    meas_std_m_     = std::max(1e-3f, meas_std_m);
    accel_std_mps2_ = std::max(1e-3f, accel_std_mps2);
    gate_nis_soft_  = std::max(0.0f, gate_nis_soft);
    gate_nis_hard_  = std::max(gate_nis_soft_, gate_nis_hard);
  }

  // score policy (可不調，用預設即可)
  void SetScorePolicy(float min_score_to_update = 0.05f,
                      float eps = 1e-3f,
                      float r_max_scale = 100.0f) {
    min_score_to_update_ = std::max(0.0f, min_score_to_update);
    score_eps_           = std::max(1e-6f, eps);
    r_max_scale_         = std::max(1.0f, r_max_scale);
  }

  // 兼容舊呼叫：沒給 score 當作 1.0
  void Update(float z_meas_dist, float dt, int frame) {
    Update(z_meas_dist, dt, frame, 1.0f);
  }

  // 新版：score 驅動 R
  void Update(float z_meas_dist, float dt, int frame, float score) {
    dt = Clamp(dt, kDtMin, kDtMax);

    if (!initialized_) {
      Initialize(z_meas_dist, dt, frame);
      last_frame_ = frame;
      return;
    }

    // 1) F(dt)
    kf_.transitionMatrix.at<float>(0, 1) = dt;

    // 2) Q(dt) (DWNA)
    UpdateProcessNoise(dt);

    // 3) Predict
    const cv::Mat pred = kf_.predict();
    const float d_pred = pred.at<float>(0);

    // 4) score -> dynamic R
    const float s = Clamp(score, 0.0f, 1.0f);
    const float r0 = meas_std_m_ * meas_std_m_;         // baseline R0

    // 太低分：視為沒有量測 => predict-only（P 會長大）
    if (s < min_score_to_update_) {
      Reject(); // keep prediction as posterior
      time_since_update_s_ += dt;
      rejection_count_ += 1;
      last_frame_ = frame;
      return;
    }

    // R_used = R0 / (score^2 + eps)
    float scale = 1.0f / (s * s + score_eps_);          // >= ~1
    scale = std::min(scale, r_max_scale_);
    const float r_used = r0 * scale;

    // 5) NIS gating 用 baseline r0（避免 score 影響 gating 邏輯）
    const float y   = z_meas_dist - d_pred;
    const float Sg  = kf_.errorCovPre.at<float>(0, 0) + r0;
    const float nis = (Sg > kEps) ? (y * y / Sg) : 0.0f;

    bool updated = false;

    if (nis <= gate_nis_soft_) {
      updated = CorrectWithR(z_meas_dist, r_used);
    } else if (nis <= gate_nis_hard_) {
      // soft update：再放大一次 R
      updated = CorrectWithR(z_meas_dist, r_used * kSoftUpdateRScale);
    } else {
      Reject();
      updated = false;
    }

    if (updated) {
      rejection_count_ = 0;
      time_since_update_s_ = 0.0f;
      last_accepted_meas_dist_ = z_meas_dist;
      has_last_accepted_meas_ = true;
      rel_speed_valid_time_s_ += dt;
      rel_speed_valid_ =
          rel_speed_valid_time_s_ >= kRelSpeedWarmupSeconds;
    } else {
      rejection_count_++;
      time_since_update_s_ += dt;

      // 這裡維持你原本的復原策略（但只有有量測時才有意義）
      if (rejection_count_ >= kMaxConsecutiveRejections ||
          time_since_update_s_ >= kMaxNoUpdateSeconds) {
        ReinitializeFromMeasurement(z_meas_dist, dt, frame);
      }
    }

    last_frame_ = frame;
  }

  bool Initialized() const { return initialized_; }

  float Distance() const { return initialized_ ? kf_.statePost.at<float>(0) : 0.0f; }
  float RelSpeed() const { return initialized_ ? kf_.statePost.at<float>(1) : 0.0f; }
  bool RelSpeedValid() const { return initialized_ && rel_speed_valid_; }

  // ✅ covariance export
  float DistanceVar() const {
    return initialized_ ? std::max(0.0f, kf_.errorCovPost.at<float>(0, 0)) : 0.0f;
  }
  float RelSpeedVar() const {
    return initialized_ ? std::max(0.0f, kf_.errorCovPost.at<float>(1, 1)) : 0.0f;
  }

  int LastFrame() const { return last_frame_; }

private:
  static constexpr float kDtMin = 0.005f;
  static constexpr float kDtMax = 0.2f;
  static constexpr float kEps   = 1e-6f;

  static constexpr float kSoftUpdateRScale = 20.0f;

  static constexpr int   kMaxConsecutiveRejections = 3;
  static constexpr float kMaxNoUpdateSeconds       = 0.3f;

  static constexpr float kRelSpeedWarmupSeconds = 0.20f;

  cv::KalmanFilter kf_;

  bool  initialized_ = false;
  int   last_frame_  = -1;

  int   rejection_count_ = 0;
  float time_since_update_s_ = 0.0f;

  bool  has_last_accepted_meas_ = false;
  float last_accepted_meas_dist_ = 0.0f;
  bool  rel_speed_valid_ = false;
  float rel_speed_valid_time_s_ = 0.0f;

  // tuning
  float meas_std_m_     = 0.6f;
  float accel_std_mps2_ = 2.0f;
  float gate_nis_soft_  = 9.0f;
  float gate_nis_hard_  = 25.0f;

  // score->R policy
  float min_score_to_update_ = 0.05f;
  float score_eps_           = 1e-3f;
  float r_max_scale_         = 100.0f;

private:
  static float Clamp(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
  }

  void Initialize(float z_meas_dist, float dt, int frame) {
    kf_.init(2, 1, 0, CV_32F);

    kf_.transitionMatrix  = (cv::Mat_<float>(2, 2) << 1, dt, 0, 1);
    kf_.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);

    kf_.statePost    = (cv::Mat_<float>(2, 1) << z_meas_dist, 0.0f);
    kf_.errorCovPost = cv::Mat::eye(2, 2, CV_32F) * 10.0f;

    kf_.processNoiseCov     = cv::Mat::eye(2, 2, CV_32F);
    kf_.measurementNoiseCov = (cv::Mat_<float>(1, 1) << (meas_std_m_ * meas_std_m_));

    initialized_ = true;
    last_frame_  = frame;

    rejection_count_ = 0;
    time_since_update_s_ = 0.0f;

    has_last_accepted_meas_ = true;
    last_accepted_meas_dist_ = z_meas_dist;
    rel_speed_valid_ = false;
    rel_speed_valid_time_s_ = 0.0f;
  }

  void UpdateProcessNoise(float dt) {
    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;
    const float sa2 = accel_std_mps2_ * accel_std_mps2_;

    kf_.processNoiseCov = (cv::Mat_<float>(2, 2) <<
      0.25f * dt4 * sa2, 0.5f * dt3 * sa2,
      0.5f  * dt3 * sa2,        dt2 * sa2
    );
  }

  bool CorrectWithR(float z_meas_dist, float r_used) {
    kf_.measurementNoiseCov.at<float>(0, 0) = std::max(kEps, r_used);

    cv::Mat_<float> measurement(1, 1);
    measurement(0) = z_meas_dist;

    kf_.correct(measurement);
    return true;
  }

  void Reject() {
    kf_.statePost    = kf_.statePre.clone();
    kf_.errorCovPost = kf_.errorCovPre.clone();
  }

  void ReinitializeFromMeasurement(float z_meas_dist, float dt, int frame) {
    (void)dt;
    const float v0 = 0.0f;

    kf_.statePost.at<float>(0) = z_meas_dist;
    kf_.statePost.at<float>(1) = v0;

    kf_.errorCovPost = cv::Mat::eye(2, 2, CV_32F);
    kf_.errorCovPost.at<float>(0, 0) = 25.0f;
    kf_.errorCovPost.at<float>(1, 1) = 100.0f;

    rejection_count_ = 0;
    time_since_update_s_ = 0.0f;

    has_last_accepted_meas_ = true;
    last_accepted_meas_dist_ = z_meas_dist;
    rel_speed_valid_ = false;
    rel_speed_valid_time_s_ = 0.0f;

    last_frame_ = frame;
  }
};

} // namespace acc
