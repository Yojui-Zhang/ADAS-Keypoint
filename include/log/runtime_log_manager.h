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
