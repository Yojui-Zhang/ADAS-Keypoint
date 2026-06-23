#pragma once

#include "keypad.h"
#include "keypad_control.h"

namespace adas_app {

void HandlePendingCommands(keypad::CommandSource& command_source,
                           keypad::RuntimeControlState& control_state);

}  // namespace adas_app
