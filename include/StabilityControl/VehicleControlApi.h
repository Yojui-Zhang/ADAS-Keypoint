#pragma once
#include <vector>
#include <string>

#include "SortTracking.h" // TrackingBox
#include "AccApi.h"
#include "lane_keeping.h"
#include "StabilitySupervisor.h"

namespace stability {

struct VehicleControlOptions {
  bool enable_lateral_control = true;
  bool enable_longitudinal_control = true;
  bool enable_supervisor = true;
};

VehicleControlCommand VehicleControl_Run(const std::vector<TrackingBox>& world_result,
                                        float ego_speed_mps,
                                        float dt_s,
                                        std::string* out_debug = nullptr);

VehicleControlCommand VehicleControl_RunWithOptions(const std::vector<TrackingBox>& world_result,
                                                    float ego_speed_mps,
                                                    float dt_s,
                                                    const VehicleControlOptions& options,
                                                    std::string* out_debug = nullptr);

void VehicleControl_SetStabilityConfig(const StabilityConfig& cfg);
StabilityConfig VehicleControl_GetStabilityConfig();

// IMU input (optional)
void VehicleControl_SetImu(double yaw_rate_rps, double alat_mps2);

} // namespace stability
