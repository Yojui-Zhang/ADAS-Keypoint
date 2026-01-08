#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace lane_keeping {
namespace internal {

// Fit y(x) = a2*x^2 + a1*x + a0 using least squares (SVD).
// Returns coeff = {a2, a1, a0}. Behavior is kept consistent with the original version.
bool FitQuadraticLeastSquares(const std::vector<cv::Point2f>& pts,
                             cv::Vec3d& coeff,
                             std::string& debug);

inline double PolyY(const cv::Vec3d& c, double x) {
    return c[0] * x * x + c[1] * x + c[2];
}

inline double PolyDyDx(const cv::Vec3d& c, double x) {
    return 2.0 * c[0] * x + c[1];
}

inline double PolyD2yDx2(const cv::Vec3d& c) {
    return 2.0 * c[0];
}

// Signed curvature: kappa = y'' / (1 + y'^2)^(3/2)
double CurvatureKappa(const cv::Vec3d& c, double x);

} // namespace internal
} // namespace lane_keeping
