#pragma once

#include "runtime_control_state.h"

namespace controller {

void SyncCanRuntimeState(const RuntimeControlState& state);
void ShutdownRuntimeControl(RuntimeControlState* state);

}  // namespace controller
