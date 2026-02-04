// WarningFilter.h
#pragma once
#include <algorithm>

namespace collision {

// Scalar 1D Kalman filter: x(k|k)
class Kalman1D {
public:
  void Reset(float x0 = 0.f, float p0 = 1.f) {
    x_ = x0;
    p_ = p0;
    inited_ = true;
  }

  float Update(float z, float q, float r) {
    if (!inited_) Reset(z, 1.f);

    // Predict
    p_ += q;

    // Update
    const float k = p_ / (p_ + r);
    x_ = x_ + k * (z - x_);
    p_ = (1.f - k) * p_;
    return x_;
  }

  float x() const { return x_; }

private:
  bool  inited_ = false;
  float x_ = 0.f;
  float p_ = 1.f;
};

// Kalman + hysteresis latch, for a stable boolean warning output.
class WarningFilter {
public:
  void Reset() {
    kf_.Reset(0.f, 1.f);
    latched_ = false;
  }

  // q_per_s: process noise per second, r: measurement noise
  bool Update(bool raw_warning,
              float dt_s,
              float q_per_s,
              float r,
              float on_th,
              float off_th,
              float* out_x = nullptr)
  {
    dt_s = std::clamp(dt_s, 0.001f, 1.0f);

    const float z = raw_warning ? 1.f : 0.f;
    const float x = kf_.Update(z, q_per_s * dt_s, r);
    if (out_x) *out_x = x;

    // hysteresis latch
    if (!latched_ && x >= on_th) latched_ = true;
    else if (latched_ && x <= off_th) latched_ = false;

    return latched_;
  }

private:
  Kalman1D kf_;
  bool     latched_ = false;
};

} // namespace collision

