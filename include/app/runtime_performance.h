#pragma once

#include <chrono>

namespace adas_app {

struct RuntimePerformanceMetrics {
  double fps = 0.0;
  double total_ms = 0.0;
  double input_ms = 0.0;
  double inference_ms = 0.0;
  double geometry_ms = 0.0;
  double acc_scope_ms = 0.0;
  double acc_ms = 0.0;
  double lka_ms = 0.0;
  double stability_ms = 0.0;
  double control_total_ms = 0.0;
  double behavior_ms = 0.0;
  double collision_ms = 0.0;
  double overlay_ms = 0.0;
};

using PerfClock = std::chrono::steady_clock;

double ElapsedMs(const PerfClock::time_point& start,
                 const PerfClock::time_point& end);

}  // namespace adas_app
