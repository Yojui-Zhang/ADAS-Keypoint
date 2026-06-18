#pragma once

#include "lane_keeping.h"

namespace lane_keeping {
namespace internal {

struct MpcSteerResult {
    bool valid = false;
    double steer_rad = 0.0;
    double delta_u_rad = 0.0;
};

MpcSteerResult ComputeMpcSteering(double cte_m,
                                  double heading_err_rad,
                                  double curvature_m_inv,
                                  double last_steer_rad,
                                  const ControlConfig& cfg);

} // namespace internal
} // namespace lane_keeping
