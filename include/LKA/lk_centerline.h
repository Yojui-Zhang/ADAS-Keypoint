#pragma once

#include <string>
#include <vector>

#include "lane_keeping.h"

namespace lane_keeping {
namespace internal {

// Build a single centerline TrackingBox (class_id=0, kpts in meters, x-forward, y-left).
// Mirrors the original logic:
// - Choose closest left and closest right lane line at a preview x.
// - If only one side exists, synthesize centerline by shifting by half lane width.
// - Sample N=15 points uniformly over x-range.
bool BuildCenterlineFromWorldResult(const std::vector<TrackingBox>& world_result,
                                   const ControlConfig& cfg,
                                   TrackingBox& out_centerline_box,
                                   std::string& debug);

} // namespace internal
} // namespace lane_keeping
