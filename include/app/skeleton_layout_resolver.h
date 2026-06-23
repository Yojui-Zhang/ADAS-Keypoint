#pragma once

#include "VehicleSkeletonAPI.h"
#include "system_config.h"

namespace adas_app {

vehicle_skeleton::SkeletonKptLayout ResolveSkeletonLayout(const AdasSystemConfig& runtime_cfg);

}  // namespace adas_app
