#pragma once
#include <vector>
#include <string>

#include "SortTracking.h" // TrackingBox
#include "AccApi.h"
#include "lane_keeping.h"

#include "StabilitySupervisor.h"

namespace stability {

// 封裝成「單一入口」，不改 ACC/LKA 內部：只做讀取 + Supervisor 限制
VehicleControlCommand VehicleControl_Run(const std::vector<TrackingBox>& world_result,
                                        float ego_speed_mps,
                                        float dt_s,
                                        std::string* out_debug = nullptr);

// 可選：調整監管器參數
void VehicleControl_SetStabilityConfig(const StabilityConfig& cfg);
StabilityConfig VehicleControl_GetStabilityConfig();

} // namespace stability

