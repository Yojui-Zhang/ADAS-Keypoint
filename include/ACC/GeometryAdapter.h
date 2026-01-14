#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>

// 這裡不重定義 TrackingBox，直接假設你專案已有該 struct。
// 若你需要獨立編譯，把 TrackingBox 定義搬到共用 header 再 include 即可。

namespace acc {

// 依你 GeometryFunction.cpp：class_id>0 時 kpts[0]=左下角, kpts[1]=右下角，且已是 (x_forward_m, y_left_m, conf)
template <typename TrackingBoxT>
inline bool TryGetGroundBottomCenterXY(const TrackingBoxT& tb, cv::Point2f& out_xy_m) {
  if (tb.class_id <= 0) return false;              // 排除 lane 或其他
  if (tb.World_box.size() < 2) return false;

  const auto& bl = tb.World_box[0];
  const auto& br = tb.World_box[1];

  if (std::isnan(bl.x) || std::isnan(bl.y) || std::isnan(br.x) || std::isnan(br.y)) return false;

  out_xy_m.x = 0.5f * (bl.x + br.x);  // forward (m)
  out_xy_m.y = 0.5f * (bl.y + br.y);  // left (m)
  return true;
}

} // namespace acc

