#pragma once
#include <algorithm>
#include <cstdint>

namespace acc {

// 簡潔、工程上很常用的距離/相對速度估測（constant-velocity alpha-beta filter）
class AlphaBetaFilter {
public:
  void Reset() { initialized_ = false; x_ = 0.0f; v_ = 0.0f; last_frame_ = -1; }

  void Update(float z_meas, float dt, int frame) {
    if (dt <= 1e-4f) dt = 1e-2f;

    if (!initialized_) {
      initialized_ = true;
      x_ = z_meas;
      v_ = 0.0f;
      last_frame_ = frame;
      return;
    }

    // Predict
    float x_pred = x_ + v_ * dt;
    float v_pred = v_;

    // Residual
    float r = z_meas - x_pred;

    // Gains (可視雜訊調整；這組對30FPS常見距離抖動夠穩)
    constexpr float alpha = 0.75f;
    constexpr float beta  = 0.08f;

    // Correct
    x_ = x_pred + alpha * r;
    v_ = v_pred + (beta / dt) * r;

    last_frame_ = frame;
  }

  bool  Initialized() const { return initialized_; }
  float Distance() const { return x_; }         // forward distance (m)
  float RelSpeed() const { return v_; }         // d_dot = (lead - ego) m/s
  int   LastFrame() const { return last_frame_; }

private:
  bool initialized_ = false;
  float x_ = 0.0f;
  float v_ = 0.0f;
  int last_frame_ = -1;
};

} // namespace acc

