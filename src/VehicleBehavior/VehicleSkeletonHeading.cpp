#include "VehicleSkeletonHeading.h"
#include <cmath>
#include <vector>
#include <limits>

namespace vehicle_skeleton {

static inline bool isFinite2D(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

static inline float normalizeAngleDeg(float a) {
    // normalize to (-180, 180]
    while (a <= -180.0f) a += 360.0f;
    while (a >  180.0f)  a -= 360.0f;
    return a;
}

static bool getWorldXY(const std::vector<cv::Point3f>& kpts, int idx, cv::Point2f& out) {
    if (idx < 0 || idx >= static_cast<int>(kpts.size())) return false;
    const auto& p = kpts[idx];
    cv::Point2f xy(p.x, p.y); // p.x=forward[m], p.y=left[m]
    if (!isFinite2D(xy)) return false;
    out = xy;
    return true;
}

static bool avgPoints(const std::vector<cv::Point2f>& pts, cv::Point2f& out) {
    if (pts.empty()) return false;
    double sx = 0.0, sy = 0.0;
    for (const auto& p : pts) { sx += p.x; sy += p.y; }
    out = cv::Point2f(static_cast<float>(sx / pts.size()), static_cast<float>(sy / pts.size()));
    return true;
}

HeadingResult ComputeVehicleHeadingFromWorldKpts(
    const std::vector<cv::Point3f>& world_kpts,
    const SkeletonKptLayout& layout,
    float ego_heading_deg
) {
    HeadingResult R;

    // front center: average of (LF, RF) across available layers
    std::vector<cv::Point2f> front_samples;
    std::vector<cv::Point2f> rear_samples;

    auto push_if_ok = [&](int idx, std::vector<cv::Point2f>& dst){
        cv::Point2f p;
        if (getWorldXY(world_kpts, idx, p)) dst.push_back(p);
    };

    // front: LF/RF for top/mid/bot
    push_if_ok(layout.top_lf, front_samples); push_if_ok(layout.top_rf, front_samples);
    push_if_ok(layout.mid_lf, front_samples); push_if_ok(layout.mid_rf, front_samples);
    push_if_ok(layout.bot_lf, front_samples); push_if_ok(layout.bot_rf, front_samples);

    // rear: LR/RR for top/mid/bot
    push_if_ok(layout.top_lr, rear_samples); push_if_ok(layout.top_rr, rear_samples);
    push_if_ok(layout.mid_lr, rear_samples); push_if_ok(layout.mid_rr, rear_samples);
    push_if_ok(layout.bot_lr, rear_samples); push_if_ok(layout.bot_rr, rear_samples);

    if (!avgPoints(front_samples, R.front_center_w) || !avgPoints(rear_samples, R.rear_center_w)) {
        R.valid = false;
        return R;
    }

    cv::Point2f v = R.front_center_w - R.rear_center_w; // vehicle longitudinal axis in world
    float norm = std::sqrt(v.x*v.x + v.y*v.y);
    if (!std::isfinite(norm) || norm < 1e-4f) {
        R.valid = false;
        return R;
    }

    float yaw = std::atan2(v.y, v.x) * 180.0f / static_cast<float>(M_PI); // +Y(left) => positive CCW
    yaw = normalizeAngleDeg(yaw - ego_heading_deg);

    R.heading_deg = yaw;
    R.valid = true;
    return R;
}

} // namespace vehicle_skeleton

