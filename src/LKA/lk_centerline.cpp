#include "lk_centerline.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "lk_lane_points.h"

namespace lane_keeping {
namespace internal {

bool BuildCenterlineFromWorldResult(const std::vector<TrackingBox>& world_result,
                                   const ControlConfig& cfg,
                                   TrackingBox& out_centerline_box,
                                   std::string& debug)
{
    // NOTE: This constant is kept identical to the original implementation.
    const float lane_width_m = 3.5f;
    const float half_lane_m  = lane_width_m * 0.5f;

    const float x_eval = std::max(0.5f, std::min(cfg.x_heading_straight_m, 3.0f));

    bool has_left  = false;
    bool has_right = false;

    std::vector<cv::Point2f> best_left_pts;
    std::vector<cv::Point2f> best_right_pts;

    float best_left_abs_y  = 1e9f;
    float best_right_abs_y = 1e9f;

    for (const auto& box : world_result) {
        if (box.class_id != 0) continue;

        std::vector<cv::Point2f> pts;
        std::string one_debug;
        const LanePointStatus status = ExtractLanePointsVehicleM(box, cfg, pts, &one_debug);
        if (status != LanePointStatus::kOk) {
            continue;
        }

        float y_eval = 0.0f;
        if (!EstimateLaneYAtX(pts, x_eval, y_eval)) {
            continue;
        }

        const float abs_y = std::fabs(y_eval);

        // Left: y > 0; Right: y < 0
        if (y_eval > 0.0f) {
            if (!has_left || abs_y < best_left_abs_y) {
                has_left = true;
                best_left_abs_y = abs_y;
                best_left_pts = std::move(pts);
            }
        } else if (y_eval < 0.0f) {
            if (!has_right || abs_y < best_right_abs_y) {
                has_right = true;
                best_right_abs_y = abs_y;
                best_right_pts = std::move(pts);
            }
        } else {
            // y == 0 is rare; ignore (kept consistent with original behavior).
        }
    }

    if (!has_left && !has_right) {
        return false;
    }

    float x_min = 0.0f;
    float x_max = 0.0f;

    if (has_left && has_right) {
        x_min = std::max(best_left_pts.front().x,  best_right_pts.front().x);
        x_max = std::min(best_left_pts.back().x,   best_right_pts.back().x);
    } else if (has_left) {
        x_min = best_left_pts.front().x;
        x_max = best_left_pts.back().x;
    } else {
        x_min = best_right_pts.front().x;
        x_max = best_right_pts.back().x;
    }

    x_min = std::max(x_min, cfg.min_x_m);
    x_max = std::min(x_max, cfg.max_x_m);

    if (x_max - x_min < 0.3f) {
        debug = "A: lane range too small to build centerline.";
        return false;
    }

    const int kNumSamples = 15; // kept identical
    std::vector<cv::Point3f> center_kpts;
    center_kpts.reserve(kNumSamples);

    for (int i = 0; i < kNumSamples; ++i) {
        const float t = (kNumSamples == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(kNumSamples - 1);
        const float xq = x_min + t * (x_max - x_min);

        float y_center = 0.0f;

        if (has_left && has_right) {
            float yL = 0.0f;
            float yR = 0.0f;
            const bool okL = SampleYLinear(best_left_pts,  xq, yL);
            const bool okR = SampleYLinear(best_right_pts, xq, yR);
            if (!okL || !okR) continue;

            y_center = 0.5f * (yL + yR);
        } else if (has_left) {
            float yL = 0.0f;
            if (!SampleYLinear(best_left_pts, xq, yL)) continue;
            y_center = yL - half_lane_m;
        } else {
            float yR = 0.0f;
            if (!SampleYLinear(best_right_pts, xq, yR)) continue;
            y_center = yR + half_lane_m;
        }

        center_kpts.emplace_back(xq, y_center, 1.0f);
    }

    if (center_kpts.size() < 3) {
        debug = "A: centerline kpts < 3 after sampling.";
        return false;
    }

    out_centerline_box = TrackingBox{};
    out_centerline_box.class_id = 0;
    out_centerline_box.kpts = std::move(center_kpts);

    std::ostringstream oss;
    oss << "A: has_left=" << has_left
        << " has_right=" << has_right
        << " | x_range=[" << x_min << "," << x_max << "]"
        << " | center_pts=" << out_centerline_box.kpts.size()
        << " | eval_x=" << x_eval
        << " | best_left_abs_y=" << (has_left ? best_left_abs_y : -1)
        << " | best_right_abs_y=" << (has_right ? best_right_abs_y : -1);

    debug = oss.str();
    return true;
}

} // namespace internal
} // namespace lane_keeping
