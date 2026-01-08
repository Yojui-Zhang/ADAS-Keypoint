#include "lk_visualization.h"

#include <vector>

#include <opencv2/imgproc.hpp>

namespace lane_keeping {
namespace internal {

void DrawCenterlineOnImage(const TrackingBox& center_lane,
                           cv::Mat& output_img,
                           const CameraModel& cam)
{
    if (output_img.empty()) return;

    std::vector<cv::Point> draw_pts;
    draw_pts.reserve(center_lane.kpts.size());

    for (const auto& kp : center_lane.kpts) {
        // center_lane.kpts: meters, X=forward, Y=left

        // Inverse transform (meter -> cm, vehicle frame -> raw world frame)
        // x_forward_m = p.y_raw_cm / 100.0  =>  p.y_raw_cm = x_forward_m * 100.0
        // y_left_m    = -p.x_raw_cm / 100.0 =>  p.x_raw_cm = -y_left_m * 100.0
        const float raw_x_cm = -kp.y * 100.0f;
        const float raw_y_cm =  kp.x * 100.0f;
        const float raw_z_cm = 0.0f; // assume ground plane

        const cv::Point2f uv = cam.project3dToPixel(cv::Point3f(raw_x_cm, raw_y_cm, raw_z_cm));
        if (uv.x >= 0 && uv.x < output_img.cols && uv.y >= 0 && uv.y < output_img.rows) {
            // Keep implicit OpenCV conversion (float -> int) consistent with the original code.
            draw_pts.push_back(uv);
        }
    }

    if (draw_pts.size() >= 2) {
        cv::polylines(output_img, draw_pts, false, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
}

} // namespace internal
} // namespace lane_keeping
