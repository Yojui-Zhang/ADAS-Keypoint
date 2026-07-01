#pragma once

#include "runtime_control_state.h"
#include "keypad.h"

namespace adas_app {

void HandlePendingCommands(keypad::CommandSource& command_source,
                           controller::RuntimeControlState& control_state);

}  // namespace adas_app
