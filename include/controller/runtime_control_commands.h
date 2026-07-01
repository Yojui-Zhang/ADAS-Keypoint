#pragma once

#include <string>

#include "runtime_control_state.h"
#include "user_command.h"

namespace controller {

bool HandleCommand(user_command_mode_t command,
                   RuntimeControlState* state,
                   std::string* out_message = nullptr);

}  // namespace controller
