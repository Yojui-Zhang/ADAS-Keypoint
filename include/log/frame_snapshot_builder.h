#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "lane_keeping.h"
#include "runtime_log_manager.h"
#include "runtime_performance.h"

namespace adas_log {

struct FrameSnapshotBuilderInput {
  uint64_t frame_index = 0;
  uint64_t frame_sync_ns = 0;
  uint64_t frame_hw_ns = 0;
  uint64_t cmd_sync_ns = 0;

  double dt_s = 0.0;
  double ego_speed_kmh = 0.0;
  double target_speed_kmh = 0.0;
  double target_distance_m = 0.0;
  double target_ttc_s = 0.0;

  const LkaReferenceSnapshot* lka_reference_snapshot = nullptr;
  bool lka_current_image_valid = false;
  cv::Point2f lka_current_px;
  bool lka_target_image_valid = false;
  cv::Point2f lka_target_px;

  const std::vector<TrackingBox>* tracking_result = nullptr;
  const std::vector<TrackingBox>* world_before_behavior = nullptr;
  const std::vector<TrackingBox>* world_result = nullptr;
  const stability::VehicleControlCommand* vehicle_cmd = nullptr;
  const collision::CollisionAssistOutput* collision_output = nullptr;

  bool can_valid = false;
  const CAR* can_state = nullptr;

  const adas_app::RuntimePerformanceMetrics* perf = nullptr;
};

FrameSnapshot BuildFrameSnapshot(const FrameSnapshotBuilderInput& input);

}  // namespace adas_log
