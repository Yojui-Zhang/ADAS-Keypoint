#include "keypad_command_dispatch.h"

#include <iostream>
#include <string>

#include "runtime_control_commands.h"
#include "runtime_control_runtime.h"
#include "terminal.h"

namespace adas_app {

void HandlePendingCommands(keypad::CommandSource& command_source,
                           controller::RuntimeControlState& control_state) {
  while (true) {
    const user_command_mode_t cmd = command_source.Consume();
    if (cmd == CMD_NONE) {
      break;
    }

    std::string message;
    if (controller::HandleCommand(cmd, &control_state, &message)) {
      controller::SyncCanRuntimeState(control_state);
      if (message.empty() == false) {
        std::cout << "Main: " << message << std::endl;
      }
    }
  }
}

}  // namespace adas_app
