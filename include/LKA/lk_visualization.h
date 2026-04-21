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

// Draw closest left/right lane curves (quadratic fit) on output image.
// The fitted model is y(x) = a2*x^2 + a1*x + a0 in vehicle ground frame (meters): x-forward, y-left.
//
// Notes:
// - This is a visualization helper only; it re-fits from world_result per call.
// - Lane selection uses the fitted y at x_eval (derived from cfg.x_heading_straight_m) and chooses
//   the nearest lane on each side (left: y>0, right: y<0).
void DrawFittedLeftRightLanesOnImage(const std::vector<TrackingBox>& world_result,
                                    cv::Mat& output_img,
                                    const CameraModel& cam,
                                    const ControlConfig& cfg);

// Draw the selected left/right lane solutions and the derived centerline using
// only the current frame's valid keypoint x positions within the requested
// forward range. This avoids extrapolating fitted curves beyond detected points.
void DrawLkaLaneSolutionOnImage(const std::vector<TrackingBox>& world_result,
                                cv::Mat& output_img,
                                const CameraModel& cam,
                                const ControlConfig& cfg,
                                float x_start_m = 0.0f,
                                float x_end_m = 20.0f);

void DrawLaneDetectOverlayOnImage(const std::vector<TrackingBox>& world_result,
                                  cv::Mat& output_img,
                                  const CameraModel& cam,
                                  const ControlConfig& cfg);

} // namespace internal
} // namespace lane_keeping
