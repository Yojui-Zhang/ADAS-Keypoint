#include "runtime_control_overlay.h"

#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "config.h"

namespace controller {
namespace {

std::string ToggleText(bool enabled) {
  return enabled ? "ON" : "OFF";
}

}  // namespace

void DrawRuntimeStatusOverlay(cv::Mat& frame,
                              const RuntimeControlState& state,
                              bool evdev_ready) {
  if (frame.empty() || state.draw_status_hud == false) {
    return;
  }

  std::vector<std::string> lines;
  lines.emplace_back("Hotkeys 1:TX 2/+:Throttle -/B:Brake 3:Steer 4:Infer 5:ACC 6:LKA 7:Behavior 8:Collision");
  lines.emplace_back("Hotkeys 9:HUD G:Grid H:LaneDet D:Demo Q:Lat W:Long E:Sup R:LDW 0:All Backspace:SafeOff");
  lines.emplace_back("CAN compile:" + ToggleText(state.canbus_compiled));
  lines.emplace_back("keypad:" + ToggleText(evdev_ready));
  lines.emplace_back("TX master:" + ToggleText(state.can_tx_master_enable));
  lines.emplace_back("longitudinal:" + ToggleText(LongitudinalControlActive(state)));
  lines.emplace_back("longitudinal ctl:" + state.longitudinal_controller_name);
  lines.emplace_back("steer:" + ToggleText(SteeringControlActive(state)));
  lines.emplace_back("Throttle:" + ToggleText(ThrottleControlActive(state)));
  lines.emplace_back("Brake:" + ToggleText(BrakeControlActive(state)));
  lines.emplace_back("Steering:" + ToggleText(SteeringControlActive(state)));
  lines.emplace_back("Draw infer:" + ToggleText(state.draw_inference_overlay));
  lines.emplace_back("ACC:" + ToggleText(state.draw_acc_overlay));
  lines.emplace_back("LKA:" + ToggleText(state.draw_lka_overlay));
  lines.emplace_back("Behavior:" + ToggleText(state.draw_behavior_overlay));
  lines.emplace_back("Collision:" + ToggleText(state.draw_collision_overlay));
  lines.emplace_back("Grid:" + ToggleText(state.draw_ground_grid_overlay));
  lines.emplace_back("LaneDet:" + ToggleText(state.draw_lane_detect_overlay));
  lines.emplace_back("Demo:" + ToggleText(state.demo_presentation_mode));
  lines.emplace_back("Demo LatCtl:" + ToggleText(DemoLateralControlEnabled(state)));
  lines.emplace_back("Demo LongCtl:" + ToggleText(DemoLongitudinalControlEnabled(state)));
  lines.emplace_back("Demo Supervisor:" + ToggleText(DemoSupervisorEnabled(state)));
  lines.emplace_back("Demo LDW:" + ToggleText(DemoLaneDepartureWarningEnabled(state)));

  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const double scale = 0.48;
  const int thickness = 1;
  const int left = 20;
  const int top = 380;

  int y = top;
  for (const auto& line : lines) {
    cv::putText(frame,
                line,
                cv::Point(left, y),
                font,
                scale,
                BLACK,
                thickness + 1,
                cv::LINE_AA);
    cv::putText(frame,
                line,
                cv::Point(left, y),
                font,
                scale,
                WHITE,
                thickness,
                cv::LINE_AA);
    y += 22;
  }
}

}  // namespace controller
