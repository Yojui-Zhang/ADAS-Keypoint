#pragma once

#include "VehicleControlApi.h"
#include "keypad_control.h"

namespace adas_app {

float SelectActuatorSpeedTargetKmh(const keypad::RuntimeControlState& control_state,
                                   const stability::VehicleControlCommand& cmd,
                                   float ego_speed_kmh);

}  // namespace adas_app
