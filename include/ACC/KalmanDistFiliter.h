#pragma once
#include "kalman.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>

namespace acc {

// ==========================================
// 轉接器： 將 2D 的 KalmanTracker 轉接為 ACC 需要的 1D 距離/速度濾波器
// ==========================================
class AccKalmanAdapter {
public:
  AccKalmanAdapter() : initialized_(false) {}

  // 介面保持與原 AlphaBetaFilter 一致，方便替換
  void Update(float z_meas_dist, float dt, int frame) {
    // 1. 建構虛擬的測量框 (Rect)
    // 我們將 ACC 的縱向距離 (dist) 映射到 Rect 的 x 軸
    // y, w, h 給定固定值即可，因為我們不關心它們
    // 注意：KalmanTracker 內部是用中心點 (cx, cy) 運算，
    // 但 get_state 會還原回 x，所以數值是一致的。
    float dummy_y = 0.0f;
    float dummy_w = 10.0f;
    float dummy_h = 10.0f;
    cv::Rect_<float> measurement(z_meas_dist, dummy_y, dummy_w, dummy_h);

    // 2. 初始化
    if (!initialized_) {
      // 使用 KalmanTracker 的帶參建構子初始化
      tracker_ = KalmanTracker(measurement, 1.0f);
      last_smooth_dist_ = z_meas_dist;
      rel_speed_ = 0.0f;
      last_frame_ = frame;
      initialized_ = true;
      return;
    }

    // 3. 執行 Kalman 預測與更新
    tracker_.predict();
    tracker_.update(measurement, 1.0f); // score 設 1.0 即可

    // 4. 獲取平滑後的位置
    cv::Rect_<float> state = tracker_.get_state();
    float current_smooth_dist = state.x;

    // 5. 外部計算速度 (因為 KalmanTracker 沒公開 velocity 介面)
    // v = (d_new - d_old) / dt
    if (dt > 1e-4f) {
       rel_speed_ = (current_smooth_dist - last_smooth_dist_) / dt;
    }

    last_smooth_dist_ = current_smooth_dist;
    last_frame_ = frame;
  }

  bool  Initialized() const { return initialized_; }
  float Distance() const { return last_smooth_dist_; }
  float RelSpeed() const { return rel_speed_; }
  int   LastFrame() const { return last_frame_; }

private:
  KalmanTracker tracker_; // 內部包含 cv::KalmanFilter
  bool  initialized_;
  float last_smooth_dist_ = 0.0f;
  float rel_speed_ = 0.0f;
  int   last_frame_ = -1;
};



} // namespace acc