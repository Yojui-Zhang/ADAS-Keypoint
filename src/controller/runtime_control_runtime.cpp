#include "runtime_control_runtime.h"

#include "brake_control.h"
#include "steering_control.h"
#include "throttle_control.h"

namespace controller {

void SyncCanRuntimeState(const RuntimeControlState& state) {
#ifdef CANBUS__
  static bool last_throttle = false;
  static bool last_brake = false;
  static bool last_steering = false;

  const bool throttle_active = ThrottleControlActive(state);
  const bool brake_active = BrakeControlActive(state);
  const bool steering_active = SteeringControlActive(state);

  if (throttle_active != last_throttle) {
    ApplyThrottleRuntime(throttle_active, state.longitudinal_controller);
    last_throttle = throttle_active;
  }

  if (brake_active != last_brake) {
    ApplyBrakeRuntime(brake_active);
    last_brake = brake_active;
  }

  if (steering_active != last_steering) {
    ApplySteeringRuntime(steering_active);
    last_steering = steering_active;
  }
#else
  (void)state;
#endif
}

void ShutdownRuntimeControl(RuntimeControlState* state) {
  if (state == nullptr) {
    return;
  }

  state->can_tx_master_enable = false;
  state->can_throttle_enable = false;
  state->can_brake_enable = false;
  state->can_steering_enable = false;
  SyncCanRuntimeState(*state);
}

}  // namespace controller
