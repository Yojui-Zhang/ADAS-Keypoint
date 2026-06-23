#include "keypad_command_dispatch.h"

#include <iostream>
#include <string>

#include "terminal.h"

namespace adas_app {

void HandlePendingCommands(keypad::CommandSource& command_source,
                           keypad::RuntimeControlState& control_state) {
  while (true) {
    const user_command_mode_t cmd = command_source.Consume();
    if (cmd == CMD_NONE) {
      break;
    }

    std::string message;
    if (keypad::HandleCommand(cmd, &control_state, &message)) {
      keypad::SyncCanRuntimeState(control_state);
      if (message.empty() == false) {
        std::cout << "Main: " << message << std::endl;
      }
    }
  }
}

}  // namespace adas_app
