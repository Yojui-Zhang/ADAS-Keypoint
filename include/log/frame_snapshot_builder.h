#pragma once

#include <cstdint>
#include <string>
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
  double final_brake_0_10 = 0.0;
  std::string final_brake_source = "none";
  double target_distance_m = 0.0;
  double target_ttc_s = 0.0;
  int throttle_mode_code = 0;
  std::string throttle_mode_text = "disabled";
  std::string throttle_requested_mode = "disabled";
  std::string throttle_effective_mode = "disabled";
  double throttle_target_speed_kmh = 0.0;
  double throttle_current_speed_kmh = 0.0;
  double throttle_visible_target_speed_kmh = 0.0;
  double throttle_operating_speed_kmh = 0.0;
  double throttle_feedforward_pedal_v = 0.75;
  double throttle_speed_error_kmh = 0.0;
  double throttle_integral_v = 0.0;
  double throttle_desired_pedal_v = 0.75;
  double throttle_final_pedal_v = 0.75;
  double throttle_applied_pedal_v = 0.75;
  double throttle_pedal_upper_v = 3.45;
  double throttle_requested_brake_0_10 = 0.0;
  bool throttle_brake_interlock_active = false;
  double throttle_measured_dt_s = 0.0;
  bool throttle_vehicle_speed_fresh = false;
  double throttle_vehicle_speed_age_ms = 0.0;
  uint64_t throttle_vehicle_speed_timestamp_ns = 0;
  bool throttle_vehicle_acceleration_fresh = false;
  double throttle_raw_acceleration_mps2 = 0.0;
  double throttle_filtered_acceleration_mps2 = 0.0;
  double throttle_measured_jerk_mps3 = 0.0;
  double throttle_allowed_acceleration_mps2 = 0.0;
  bool throttle_acceleration_guard_active = false;
  bool throttle_jerk_guard_active = false;
  std::string throttle_calibration_id = "throttle_default_v1";
  bool brake_control_active = false;
  uint64_t acc_manual_resume_request_sequence = 0;

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
  const collision::AebAudioGateOutput* aeb_audio_gate = nullptr;

  bool can_valid = false;
  const CAR* can_state = nullptr;

  const adas_app::RuntimePerformanceMetrics* perf = nullptr;
};

FrameSnapshot BuildFrameSnapshot(const FrameSnapshotBuilderInput& input);

}  // namespace adas_log
