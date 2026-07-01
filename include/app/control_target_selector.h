#pragma once

#include "runtime_control_state.h"
#include "VehicleControlApi.h"

namespace adas_app {

float SelectActuatorSpeedTargetKmh(const controller::RuntimeControlState& control_state,
                                   const stability::VehicleControlCommand& cmd,
                                   float ego_speed_kmh);

}  // namespace adas_app
