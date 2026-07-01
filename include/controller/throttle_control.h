#pragma once

#include "runtime_control_state.h"

namespace controller {

void ApplyThrottleRuntime(bool active, LongitudinalControllerKind controller_kind);
void StopThrottleRuntime();

}  // namespace controller
