#include "CameraProjectionUtils.h"

#include <cmath>

#include "CameraModel.h"

namespace {

bool IsFinitePoint(const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

cv::Point2f ScaleProjectedPoint(const CameraModel& cam,
                                const cv::Size& image_size,
                                const cv::Point2f& point) {
    const cv::Size calib_size = cam.calibrationImageSize();
    if (calib_size.width <= 0 || calib_size.height <= 0 ||
        image_size.width <= 0 || image_size.height <= 0) {
        return point;
    }

    if (calib_size == image_size) {
        return point;
    }

    const float sx = static_cast<float>(image_size.width) /
                     static_cast<float>(calib_size.width);
    const float sy = static_cast<float>(image_size.height) /
                     static_cast<float>(calib_size.height);
    return cv::Point2f(point.x * sx, point.y * sy);
}

}  // namespace

cv::Point3f RawWorldPointFromVehicleMeters(float forward_m, float left_m) {
    return cv::Point3f(-left_m * 100.0f, forward_m * 100.0f, 0.0f);
}

cv::Point2f ProjectRawWorldPointToImage(const CameraModel& cam,
                                        const cv::Size& image_size,
                                        const cv::Point3f& raw_world_cm) {
    const cv::Mat point_w =
        (cv::Mat_<float>(3, 1) << raw_world_cm.x, raw_world_cm.y, raw_world_cm.z);
    const cv::Mat point_c = cam.Rcw() * point_w + cam.tcw();
    if (point_c.at<float>(2) <= 1e-5f) {
        return cv::Point2f(-1.0f, -1.0f);
    }

    const cv::Point2f projected = cam.project3dToPixel(raw_world_cm);
    if (IsFinitePoint(projected) == false) {
        return cv::Point2f(-1.0f, -1.0f);
    }
    return ScaleProjectedPoint(cam, image_size, projected);
}

cv::Point2f ProjectVehicleGroundPointToImage(const CameraModel& cam,
                                             const cv::Size& image_size,
                                             float forward_m,
                                             float left_m) {
    return ProjectRawWorldPointToImage(cam,
                                       image_size,
                                       RawWorldPointFromVehicleMeters(forward_m, left_m));
}

bool IsProjectedPointInsideImage(const cv::Size& image_size,
                                 const cv::Point2f& point,
                                 float padding_px) {
    if (IsFinitePoint(point) == false) {
        return false;
    }
    return point.x >= -padding_px &&
           point.y >= -padding_px &&
           point.x < static_cast<float>(image_size.width) + padding_px &&
           point.y < static_cast<float>(image_size.height) + padding_px;
}
