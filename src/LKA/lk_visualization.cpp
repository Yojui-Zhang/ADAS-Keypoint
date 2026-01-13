#include "lk_visualization.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

#include "lk_polyfit.h"
#include "lk_lane_selector.h" // [NEW] 引入選擇器

namespace lane_keeping {
namespace internal {

// (DrawCenterlineOnImage 保持不變，略過以節省篇幅...)
void DrawCenterlineOnImage(const TrackingBox& center_lane,
                           cv::Mat& output_img,
                           const CameraModel& cam) {
    // ... 原本的程式碼保持不變 ...
    if (output_img.empty()) return;
    std::vector<cv::Point> draw_pts;
    draw_pts.reserve(center_lane.kpts.size());
    for (const auto& kp : center_lane.kpts) {
        const float raw_x_cm = -kp.y * 100.0f;
        const float raw_y_cm =  kp.x * 100.0f;
        const cv::Point2f uv = cam.project3dToPixel(cv::Point3f(raw_x_cm, raw_y_cm, 0.0f));
        if (uv.x >= 0 && uv.x < output_img.cols && uv.y >= 0 && uv.y < output_img.rows) {
            draw_pts.push_back(uv);
        }
    }
    if (draw_pts.size() >= 2) {
        cv::polylines(output_img, draw_pts, false, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
}

void DrawFittedLeftRightLanesOnImage(const std::vector<TrackingBox>& world_result,
                                    cv::Mat& output_img,
                                    const CameraModel& cam,
                                    const ControlConfig& cfg)
{
    if (output_img.empty()) return;

    // 1. 使用共用模組找出最佳左右車道
    const LanePair lanes = FindBestLaneCandidates(world_result, cfg);

    if (!lanes.left.valid && !lanes.right.valid) return;

    // 2. 決定畫圖範圍 (邏輯與 Centerline 類似但只為了視覺化)
    float x_min = 0.0f;
    float x_max = 0.0f;
    if (lanes.left.valid && lanes.right.valid) {
        x_min = std::max(lanes.left.pts.front().x, lanes.right.pts.front().x);
        x_max = std::min(lanes.left.pts.back().x, lanes.right.pts.back().x);
    } else if (lanes.left.valid) {
        x_min = lanes.left.pts.front().x;
        x_max = lanes.left.pts.back().x;
    } else {
        x_min = lanes.right.pts.front().x;
        x_max = lanes.right.pts.back().x;
    }

    x_min = std::max(x_min, cfg.min_x_m);
    // x_min = cfg.min_x_m;

    // 取 (原本計算的終點) 與 (視覺限制) 的最小值
    x_max = std::min(x_max, cfg.visual_limit_m);
    if (x_max - x_min < 0.3f) return;

    const int kSamples = 60;

    // Helper: 畫單一條多項式曲線
    auto draw_poly = [&](const cv::Vec3d& c, const cv::Scalar& color, int thickness) {
        // 如果係數全為0 (表示擬合失敗但有點)，不畫曲線 (或者你可以選擇畫 raw points)
        if (c == cv::Vec3d(0,0,0)) return;

        std::vector<cv::Point> draw_pts;
        draw_pts.reserve(kSamples);

        for (int i = 0; i < kSamples; ++i) {
            const float t = (kSamples == 1) ? 0.0f
                                            : static_cast<float>(i) / static_cast<float>(kSamples - 1);
            const float x = x_min + t * (x_max - x_min);
            const float y = static_cast<float>(PolyY(c, static_cast<double>(x)));

            // 座標轉換 (vehicle meters -> raw cm)
            const float raw_x_cm = -y * 100.0f;
            const float raw_y_cm =  x * 100.0f;
            const float raw_z_cm =  0.0f;

            const cv::Point2f uv = cam.project3dToPixel(cv::Point3f(raw_x_cm, raw_y_cm, raw_z_cm));
            if (uv.x >= 0 && uv.x < output_img.cols && uv.y >= 0 && uv.y < output_img.rows) {
                draw_pts.push_back(uv);
            }
        }

        if (draw_pts.size() >= 2) {
            cv::polylines(output_img, draw_pts, false, color, thickness, cv::LINE_AA);
        }
    };

    // Left: Green, Right: Blue
    if (lanes.left.valid)  draw_poly(lanes.left.poly,  cv::Scalar(0, 255, 0), 2);
    if (lanes.right.valid) draw_poly(lanes.right.poly, cv::Scalar(255, 0, 0), 2);
}

} // namespace internal
} // namespace lane_keeping