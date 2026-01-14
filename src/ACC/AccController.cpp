#include "AccController.h"
#include <algorithm>
#include <cmath>

namespace acc {

// 注意：Clamp, KmhToMps 等已經移到 .h 成為 inline member function
// 這裡只需要實作非 Template 且較複雜的函式

float AccController::ComputeDtSec(int current_frame) {
  if (cfg_.default_fps <= 1e-3f) return 0.033f;
  if (last_frame_ < 0 || current_frame < 0) return 1.0f / cfg_.default_fps;

  int df = current_frame - last_frame_;
  if (df <= 0) return 1.0f / cfg_.default_fps;

  float dt = static_cast<float>(df) / cfg_.default_fps;
  // 調用 .h 裡的 member function
  dt = Clamp(dt, 0.005f, 0.2f);
  return dt;
}

void AccController::SetEgoSpeedKmh(float ego_speed_kmh) {

  cfg_.use_external_ego_speed = true;
  // 調用 .h 裡的 member function
  ego_speed_est_mps_ = std::max(0.0f, KmhToMps(ego_speed_kmh));
}

// Update 函式已移除，因為它現在位於 Header 檔中

} // namespace acc