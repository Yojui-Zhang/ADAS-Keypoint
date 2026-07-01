#pragma once

#include <opencv2/core.hpp>

#include "runtime_control_state.h"

namespace controller {

void DrawRuntimeStatusOverlay(cv::Mat& frame,
                              const RuntimeControlState& state,
                              bool evdev_ready);

}  // namespace controller
