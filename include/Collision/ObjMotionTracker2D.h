#pragma once
#include <unordered_map>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace collision {

// Alpha-Beta tracker for 2D relative motion (ego frame)
// - Adds: track age (warmup), residual reset (ID switch/noise), velocity clamp (spike guard)
// 目的：降低速度估測抖動/跳變，避免 TTC 在不緊急狀況被誤觸發。
struct MotionState2D {
  bool initialized = false;
  cv::Point2f pos{0.f, 0.f};        // m (ego frame): x forward, y left
  cv::Point2f vel{0.f, 0.f};        // m/s (relative)
  int last_frame = -1;
  int miss_count = 0;

  int age = 0;                       // successful update count (warmup use)
  float last_dt_s = 0.f;
};

class AlphaBetaTracker2D {
public:
  void SetGains(float alpha, float beta) {
    alpha_ = std::clamp(alpha, 0.0f, 1.0f);
    beta_  = std::max(0.0f, beta);
  }

  void SetDtClamp(float dt_min, float dt_max) {
    dt_min_ = std::max(1e-4f, dt_min);
    dt_max_ = std::max(dt_min_, dt_max);
  }

  void SetStaleFrames(int n) { stale_frames_ = std::max(1, n); }

  // 若 measurement 與 prediction 差太大，直接 reset (避免 ID swap / 瞬間跳點造成速度爆衝)
  void SetResidualResetMeters(float m) { residual_reset_m_ = std::max(0.0f, m); }

  // 速度上限 (m/s)
  void SetVelMaxMps(float vmax) { vel_max_mps_ = std::max(0.1f, vmax); }

  bool Has(int id) const { return states_.find(id) != states_.end(); }

  const MotionState2D* Get(int id) const {
    auto it = states_.find(id);
    return (it == states_.end()) ? nullptr : &it->second;
  }

  // Update (alpha-beta). z_pos is measurement in ego frame.
  MotionState2D Update(int id, int frame, const cv::Point2f& z_pos, float dt_s) {
    dt_s = std::clamp(dt_s, dt_min_, dt_max_);

    auto& st = states_[id];

    // Init
    if (!st.initialized) {
      st.initialized = true;
      st.pos = z_pos;
      st.vel = cv::Point2f(0.f, 0.f);
      st.last_frame = frame;
      st.miss_count = 0;
      st.age = 1;
      st.last_dt_s = dt_s;
      return st;
    }

    // Predict
    const cv::Point2f pred_pos = st.pos + st.vel * dt_s;
    cv::Point2f r = z_pos - pred_pos;

    // Residual reset guard
    if (residual_reset_m_ > 0.f) {
      const float r2 = r.x * r.x + r.y * r.y;
      if (r2 > residual_reset_m_ * residual_reset_m_) {
        st.pos = z_pos;
        st.vel = cv::Point2f(0.f, 0.f);
        st.last_frame = frame;
        st.miss_count = 0;
        st.age += 1;
        st.last_dt_s = dt_s;
        return st;
      }
    }

    // Update (alpha-beta)
    st.pos = pred_pos + alpha_ * r;
    st.vel = st.vel + (beta_ / std::max(1e-3f, dt_s)) * r;

    // Velocity clamp (spike guard)
    const float v2 = st.vel.x * st.vel.x + st.vel.y * st.vel.y;
    const float vmax2 = vel_max_mps_ * vel_max_mps_;
    if (v2 > vmax2) {
      const float v = std::sqrt(std::max(1e-12f, v2));
      st.vel *= (vel_max_mps_ / v);
    }

    st.last_frame = frame;
    st.miss_count = 0;
    st.age += 1;
    st.last_dt_s = dt_s;
    return st;
  }

  void MarkMissedAndPrune(const std::unordered_map<int, bool>& seen) {
    for (auto it = states_.begin(); it != states_.end(); ) {
      const int id = it->first;
      auto& st = it->second;

      auto f = seen.find(id);
      if (f == seen.end() || !f->second) st.miss_count++;

      if (st.miss_count >= stale_frames_) it = states_.erase(it);
      else ++it;
    }
  }

  void Reset() { states_.clear(); }

private:
  std::unordered_map<int, MotionState2D> states_;

  float alpha_ = 0.6f;
  float beta_  = 0.2f;

  float dt_min_ = 0.005f;
  float dt_max_ = 0.2f;

  int stale_frames_ = 10;

  float residual_reset_m_ = 4.0f; // meters
  float vel_max_mps_ = 50.0f;     // m/s (relative speed guard)
};

} // namespace collision
