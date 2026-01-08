#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "lane_keeping.h"  // ControlConfig, TrackingBox

namespace lane_keeping {
namespace internal {

// Extraction result to preserve original debug/branching behavior.
enum class LanePointStatus {
    kOk = 0,
    kNotLane,
    kEmpty,
    kTooFewAfterFilter
};

// Extract keypoints (already in vehicle ground frame in meters: x-forward, y-left).
// - Filters by confidence, x-range, and y-range.
// - Sorts by x ascending (near -> far).
//
// Returns LanePointStatus::kOk on success.
LanePointStatus ExtractLanePointsVehicleM(const TrackingBox& box,
                                         const ControlConfig& cfg,
                                         std::vector<cv::Point2f>& out_pts,
                                         std::string* debug = nullptr);

// Linear interpolation y(xq) over x-sorted points.
bool SampleYLinear(const std::vector<cv::Point2f>& pts,
                   float xq,
                   float& yq);

// Estimate lane y at x_eval; if x_eval is outside range, clamps to nearest endpoint.
bool EstimateLaneYAtX(const std::vector<cv::Point2f>& pts,
                      float x_eval,
                      float& y_eval);

} // namespace internal
} // namespace lane_keeping
