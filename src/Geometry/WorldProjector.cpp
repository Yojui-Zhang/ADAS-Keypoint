#include "WorldProjector.h"
#include <cmath>

WorldProjector::WorldProjector(CameraModel cam, GroundPlane plane)
: cam_(std::move(cam)), plane_(std::move(plane))
{
}

bool WorldProjector::pixelToWorldOnPlane(const cv::Point2f& px, cv::Point3f& out_w) const
{
    // 1) undistort
    cv::Point2f u = cam_.undistortPixel(px);

    // 2) ray in camera frame
    cv::Vec3f dir_c = cam_.pixelToUnitRayCamera(u);

    // 3) transform ray to world frame
    cv::Mat Rwc = cam_.Rwc();                 // 3x3
    cv::Mat Cw_m = cam_.cameraCenterWorld();  // 3x1

    cv::Vec3f Cw(Cw_m.at<float>(0), Cw_m.at<float>(1), Cw_m.at<float>(2));

    cv::Mat dir_c_m = (cv::Mat_<float>(3,1) << dir_c[0], dir_c[1], dir_c[2]);
    cv::Mat dir_w_m = Rwc * dir_c_m;
    cv::Vec3f dir_w(dir_w_m.at<float>(0), dir_w_m.at<float>(1), dir_w_m.at<float>(2));

    // 4) intersect ray with plane: X = C + s*dir
    // plane: n^T X + d = 0
    const cv::Vec3f& n = plane_.n();
    float d = plane_.d();

    float denom = n.dot(dir_w);
    if (std::fabs(denom) < 1e-8f) {
        out_w = NaNPoint();
        return false; // parallel
    }

    float s = -(n.dot(Cw) + d) / denom;

    // if s < 0 -> intersection behind camera
    if (s <= 0.0f) {
        out_w = NaNPoint();
        return false;
    }

    cv::Vec3f Xw = Cw + s * dir_w;
    out_w = cv::Point3f(Xw[0], Xw[1], Xw[2]);
    return true;
}
