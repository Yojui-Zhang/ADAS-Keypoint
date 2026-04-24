#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "CollisionAssistApi.h"
#include "VehicleControlApi.h"
#include "algorithm_ablation_logger.h"
#include "canbus_recv.h"
#include "research_data_logger.h"
#include "system_config.h"

namespace adas_log {

struct FrameSnapshot {
  uint64_t frame_index = 0;
  uint64_t frame_sync_ns = 0;
  uint64_t frame_hw_ns = 0;
  uint64_t cmd_sync_ns = 0;

  double dt_s = 0.0;
  double ego_speed_kmh = 0.0;
  double target_speed_kmh = 0.0;
  double target_distance_m = 0.0;
  double target_ttc_s = 0.0;

  double perf_fps = 0.0;
  double perf_total_ms = 0.0;
  double perf_input_ms = 0.0;
  double perf_inference_ms = 0.0;
  double perf_geometry_ms = 0.0;
  double perf_acc_scope_ms = 0.0;
  double perf_acc_ms = 0.0;
  double perf_lka_ms = 0.0;
  double perf_stability_ms = 0.0;
  double perf_control_total_ms = 0.0;
  double perf_behavior_ms = 0.0;
  double perf_collision_ms = 0.0;
  double perf_overlay_ms = 0.0;

  bool lka_reference_valid = false;
  double lka_p_curve = 0.0;
  double lka_ey_m = 0.0;
  double lka_epsi_rad = 0.0;
  double lka_mean_kappa_m_inv = 0.0;
  double lka_std_kappa_m_inv = 0.0;
  double lka_current_x_m = 0.0;
  double lka_current_y_m = 0.0;
  bool lka_current_image_valid = false;
  double lka_current_u_px = 0.0;
  double lka_current_v_px = 0.0;
  double lka_target_x_m = 0.0;
  double lka_target_y_m = 0.0;
  bool lka_target_image_valid = false;
  double lka_target_u_px = 0.0;
  double lka_target_v_px = 0.0;

  const std::vector<TrackingBox>* tracking_result = nullptr;
  const std::vector<TrackingBox>* world_before_behavior = nullptr;
  const std::vector<TrackingBox>* world_result = nullptr;
  const stability::VehicleControlCommand* vehicle_cmd = nullptr;
  const collision::CollisionAssistOutput* collision_output = nullptr;

  bool can_valid = false;
  const CAR* can_state = nullptr;
};

class RuntimeLogManager {
public:
  RuntimeLogManager(const AdasSystemConfig& runtime_cfg,
                    const std::string& cfg_path);
  ~RuntimeLogManager();

  bool Start(bool enable_research_logger,
             std::string* out_error = nullptr);
  bool RunVirtualRoadSimulation(std::string* out_error = nullptr);
  void LogFrame(const FrameSnapshot& snapshot);
  void Stop();

  bool AblationRunning() const { return ablation_logger_.IsRunning(); }
  bool ResearchRunning() const { return research_logger_.IsRunning(); }
  const std::string& AblationOutputPath() const { return ablation_logger_.OutputPath(); }
  const std::string& ResearchOutputPath() const { return research_logger_.OutputPath(); }

private:
  AdasSystemConfig runtime_cfg_;
  std::string cfg_path_;
  ablation::AlgorithmAblationLogger ablation_logger_;
  ResearchDataLogger research_logger_;
};

}  // namespace adas_log
