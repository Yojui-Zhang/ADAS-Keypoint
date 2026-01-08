#pragma once

#include <string>

#include "lane_keeping.h"

namespace lane_keeping {
namespace internal {

// Internal helpers (kept in a separate module to reduce clutter in the public entrypoint).
double StanleyFeedback(double cte_m,
                       double heading_err_rad,
                       double v_mps,
                       double k,
                       double softening);

double MetricToProbability(double metric,
                           const ControlConfig& cfg,
                           bool prev_mode_curve,
                           bool& out_mode_curve);

} // namespace internal
} // namespace lane_keeping
