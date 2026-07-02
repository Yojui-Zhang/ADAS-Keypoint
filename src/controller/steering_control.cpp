#include "steering_control.h"

#include "config.h"

#ifdef CANBUS__
#include "canbus_recv.h"
#endif

namespace controller {

void ApplySteeringRuntime(bool active) {
#ifdef CANBUS__
  canbus_set_steering_tx_enabled(active ? 1 : 0);
#else
  (void)active;
#endif
}

}  // namespace controller
