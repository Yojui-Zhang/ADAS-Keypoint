#pragma once

#include "AccConfig.h"
#include "GeometryAdapter.h"   // TryGetGroundBottomCenterXY
#include "config.h"           // TrackingBox (依你專案實際路徑調整)

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace acc {

// 在影像上繪製追蹤方框與資訊：
// - 會高亮 cmd.target_id（若存在）
// - 文字顯示 id / cls / distance(m)；target 額外顯示 TTC / lead speed
inline void ACC_DrawTrackingBoxes(cv::Mat& frame,
                                 const std::vector<TrackingBox>& world_result,
                                 const AccCommand& cmd,
                                 int thickness = 2) {
  if (frame.empty()) return;

  for (const auto& tb : world_result) {
    // 只畫 cls 1/2/3（如需全畫可移除此條件）
    if (!(tb.class_id == 1 || tb.class_id == 2 || tb.class_id == 3)) continue;

    const bool is_target = (tb.id == cmd.target_id);
    const cv::Scalar color = is_target ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

    // tb.box 在你的 GeometryFunction.cpp 流程仍是 pixel bbox，可直接畫
    cv::rectangle(frame, tb.box, color, thickness);

    // 估測距離（world kpts 取 bottom-center）
    cv::Point2f ground_xy;
    float dist_m = 0.0f;
    if (TryGetGroundBottomCenterXY(tb, ground_xy)) dist_m = ground_xy.x;

    std::ostringstream oss;
    oss << "id:" << tb.id << " cls:" << tb.class_id
        << " d:" << std::fixed << std::setprecision(1) << dist_m << "m";

    if (is_target) {
      if (std::isfinite(cmd.TargetTTC)) {
        oss << " TTC:" << std::fixed << std::setprecision(1) << cmd.TargetTTC << "s";
      } else {
        oss << " TTC:inf";
      }
      oss << " v_lead:" << std::fixed << std::setprecision(1) << cmd.TargetSpeedKmh << "km/h";
    }

    cv::Point org(tb.box.x, std::max(0, tb.box.y - 6));
    const std::string text = oss.str();

    cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
    cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }
}

} // namespace acc

