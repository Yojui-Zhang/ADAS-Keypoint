#include "lk_polyfit.h"

#include <cmath>
#include <sstream>

namespace lane_keeping {
namespace internal {

bool FitQuadraticLeastSquares(const std::vector<cv::Point2f>& pts,
                             cv::Vec3d& coeff,
                             std::string& debug)
{
    if (pts.size() < 3) {
        debug = "fit_quadratic: need >= 3 pts.";
        return false;
    }

    const int n = static_cast<int>(pts.size());
    cv::Mat A(n, 3, CV_64F);
    cv::Mat b(n, 1, CV_64F);

    for (int i = 0; i < n; ++i) {
        const double x = pts[i].x;
        const double y = pts[i].y;
        A.at<double>(i, 0) = x * x;
        A.at<double>(i, 1) = x;
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = y;
    }

    cv::Mat sol;
    const bool ok = cv::solve(A, b, sol, cv::DECOMP_SVD);
    if (!ok || sol.rows != 3) {
        debug = "fit_quadratic: cv::solve failed.";
        return false;
    }

    coeff[0] = sol.at<double>(0, 0);
    coeff[1] = sol.at<double>(1, 0);
    coeff[2] = sol.at<double>(2, 0);

    std::ostringstream oss;
    oss << "poly a2=" << coeff[0] << " a1=" << coeff[1] << " a0=" << coeff[2];
    debug = oss.str();
    return true;
}

double CurvatureKappa(const cv::Vec3d& c, double x) {
    const double yp  = PolyDyDx(c, x);
    const double ypp = PolyD2yDx2(c);

    const double denom = std::pow(1.0 + yp * yp, 1.5);
    if (denom < 1e-9) return 0.0;
    return ypp / denom;
}

} // namespace internal
} // namespace lane_keeping
