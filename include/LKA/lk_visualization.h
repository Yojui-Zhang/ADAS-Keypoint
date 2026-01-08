#pragma once

#include <opencv2/core.hpp>

#include "lane_keeping.h"

namespace lane_keeping {
namespace internal {

// Draw centerline points on output image by projecting back to pixel coordinates.
// Behavior is kept identical to the original in lane_steering_step.
void DrawCenterlineOnImage(const TrackingBox& center_lane,
                           cv::Mat& output_img,
                           const CameraModel& cam);

} // namespace internal
} // namespace lane_keeping
