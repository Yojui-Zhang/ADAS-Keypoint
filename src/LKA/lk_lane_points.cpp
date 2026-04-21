#include "lk_lane_points.h"

#include <algorithm>
#include <cstddef>
#include <cmath>

namespace lane_keeping {
namespace internal {

namespace {
constexpr std::size_t kMaxLaneKeypointsPerDetection = 15;
}  // namespace

LanePointStatus ExtractLanePointsVehicleM(const TrackingBox& box,
                                         const ControlConfig& cfg,
                                         std::vector<cv::Point2f>& out_pts,
                                         std::string* debug)
{
    out_pts.clear();

    if (box.class_id != 0) {
        if (debug) *debug = "not lane class_id(0).";
        return LanePointStatus::kNotLane;
    }
    if (box.kpts.empty()) {
        if (debug) *debug = "empty kpts.";
        return LanePointStatus::kEmpty;
    }

    const std::size_t kpt_count =
        std::min(box.kpts.size(), kMaxLaneKeypointsPerDetection);
    for (std::size_t i = 0; i < kpt_count; ++i) {
        const auto& kp = box.kpts[i];
        if (cfg.use_confidence && kp.z < cfg.conf_threshold) continue;

        const float x = kp.x;
        const float y = kp.y;

        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        if (x < cfg.min_x_m || x > cfg.max_x_m) continue;
        if (std::fabs(y) > cfg.max_abs_y_m) continue;

        out_pts.emplace_back(x, y);
    }

    if (out_pts.size() < 3) {
        if (debug) *debug = "valid pts < 3 after filtering.";
        return LanePointStatus::kTooFewAfterFilter;
    }

    std::sort(out_pts.begin(), out_pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    if (debug) {
        *debug = "ok pts=" + std::to_string(out_pts.size()) +
                 " used_raw_kpts=" + std::to_string(kpt_count);
    }
    return LanePointStatus::kOk;
}

bool SampleYLinear(const std::vector<cv::Point2f>& pts,
                   float xq,
                   float& yq)
{
    if (pts.size() < 2) return false;
    if (xq < pts.front().x || xq > pts.back().x) return false;

    auto it = std::lower_bound(
        pts.begin(), pts.end(), xq,
        [](const cv::Point2f& p, float value) { return p.x < value; });

    if (it == pts.begin()) {
        yq = it->y;
        return true;
    }
    if (it == pts.end()) {
        yq = pts.back().y;
        return true;
    }

    const auto& p2 = *it;
    const auto& p1 = *(it - 1);

    const float dx = p2.x - p1.x;
    if (std::fabs(dx) < 1e-6f) {
        yq = p2.y;
        return true;
    }

    const float t = (xq - p1.x) / dx;
    yq = p1.y + t * (p2.y - p1.y);
    return true;
}

bool EstimateLaneYAtX(const std::vector<cv::Point2f>& pts,
                      float x_eval,
                      float& y_eval)
{
    if (pts.empty()) return false;

    if (x_eval <= pts.front().x) {
        y_eval = pts.front().y;
        return true;
    }
    if (x_eval >= pts.back().x) {
        y_eval = pts.back().y;
        return true;
    }
    return SampleYLinear(pts, x_eval, y_eval);
}

} // namespace internal
} // namespace lane_keeping
