#include "brake_control.h"

#include "config.h"

#ifdef CANBUS__
#include "canbus_recv.h"

extern volatile double deceleration;
#endif

namespace controller {
namespace {

#ifdef CANBUS__
bool& BrakeSenderStarted() {
  static bool started = false;
  return started;
}
#endif

}  // namespace

void ApplyBrakeRuntime(bool active) {
#ifdef CANBUS__
  bool& brake_sender_started = BrakeSenderStarted();
  if (active) {
    if (brake_sender_started == false) {
      canbus_ctrl_dec(1);
      brake_sender_started = true;
    }
    return;
  }

  ::deceleration = 0.0;
  if (brake_sender_started) {
    canbus_ctrl_dec(0);
    brake_sender_started = false;
  }
#else
  (void)active;
#endif
}

}  // namespace controller
