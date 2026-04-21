#include "lk_centerline.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <sstream>

#include "lk_lane_points.h" // SampleYLinear
#include "lk_polyfit.h"     // PolyY
#include "lk_lane_selector.h" // [NEW] 引入選擇器

namespace lane_keeping {
namespace internal {

namespace {

constexpr std::size_t kMaxCenterlineKeypoints = 15;

bool IsPolyUsable(const cv::Vec3d& poly) {
    return poly != cv::Vec3d(0.0, 0.0, 0.0);
}

std::vector<float> DownsampleXsToMax(const std::vector<float>& xs,
                                     std::size_t max_count) {
    if (xs.size() <= max_count || max_count == 0) {
        return xs;
    }

    std::vector<float> out;
    out.reserve(max_count);
    for (std::size_t i = 0; i < max_count; ++i) {
        const double t = (max_count == 1)
                             ? 0.0
                             : static_cast<double>(i) / static_cast<double>(max_count - 1);
        const std::size_t idx = static_cast<std::size_t>(
            std::llround(t * static_cast<double>(xs.size() - 1)));
        out.push_back(xs[std::min(idx, xs.size() - 1)]);
    }
    return out;
}

std::vector<float> CollectPointXsInRange(const std::vector<cv::Point2f>& pts,
                                         float x_min,
                                         float x_max) {
    std::vector<float> xs;
    xs.reserve(std::min<std::size_t>(pts.size(), kMaxCenterlineKeypoints));

    for (const auto& p : pts) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        if (p.x < x_min || p.x > x_max) {
            continue;
        }
        if (xs.empty() || std::fabs(p.x - xs.back()) > 1e-3f) {
            xs.push_back(p.x);
        }
    }

    return DownsampleXsToMax(xs, kMaxCenterlineKeypoints);
}

std::vector<float> ChooseCenterlineSampleXs(const LanePair& lanes,
                                            bool has_left,
                                            bool has_right,
                                            float x_min,
                                            float x_max) {
    const std::vector<float> left_xs =
        has_left ? CollectPointXsInRange(lanes.left.pts, x_min, x_max) : std::vector<float>{};
    const std::vector<float> right_xs =
        has_right ? CollectPointXsInRange(lanes.right.pts, x_min, x_max) : std::vector<float>{};

    if (has_left && has_right) {
        if (left_xs.size() >= 3 && right_xs.size() >= 3) {
            return (left_xs.size() <= right_xs.size()) ? left_xs : right_xs;
        }
        if (left_xs.size() >= 3) {
            return left_xs;
        }
        return right_xs;
    }

    return has_left ? left_xs : right_xs;
}

bool SampleLaneYWithinKeypointRange(const LaneCandidate& lane,
                                    bool poly_ok,
                                    float xq,
                                    float& yq) {
    if (!lane.valid || lane.pts.size() < 2) {
        return false;
    }
    if (xq < lane.pts.front().x || xq > lane.pts.back().x) {
        return false;
    }
    if (poly_ok) {
        yq = static_cast<float>(PolyY(lane.poly, xq));
        return true;
    }
    return SampleYLinear(lane.pts, xq, yq);
}

}  // namespace

bool BuildCenterlineFromWorldResult(const std::vector<TrackingBox>& world_result,
                                   const ControlConfig& cfg,
                                   TrackingBox& out_centerline_box,
                                   std::string& debug)
{
    // 1. 使用共用模組找出最佳左右車道
    const LanePair lanes = FindBestLaneCandidates(world_result, cfg);
    const bool has_left = lanes.left.valid;
    const bool has_right = lanes.right.valid;

    if (!has_left && !has_right) {
        return false;
    }

    // 2. 計算取樣範圍 x_min, x_max
    float x_min = 0.0f;
    float x_max = 0.0f;

    if (has_left && has_right) {
        x_min = std::max(lanes.left.pts.front().x,  lanes.right.pts.front().x);
        x_max = std::min(lanes.left.pts.back().x,   lanes.right.pts.back().x);
    } else if (has_left) {
        x_min = lanes.left.pts.front().x;
        x_max = lanes.left.pts.back().x;
    } else {
        x_min = lanes.right.pts.front().x;
        x_max = lanes.right.pts.back().x;
    }

    x_min = std::max(x_min, cfg.min_x_m);
    x_max = std::min(x_max, cfg.max_x_m);

    if (x_max - x_min < 0.3f) {
        debug = "A: lane range too small to build centerline.";
        return false;
    }

    // 3. 只用當幀有效 keypoints 的 x 位置生成中心線，不再固定補滿 15 點。
    const std::vector<float> sample_xs =
        ChooseCenterlineSampleXs(lanes, has_left, has_right, x_min, x_max);
    if (sample_xs.size() < 3) {
        debug = "A: centerline usable keypoint xs < 3.";
        return false;
    }

    std::vector<cv::Point3f> center_kpts;
    center_kpts.reserve(sample_xs.size());
    
    const bool left_poly_ok = has_left && IsPolyUsable(lanes.left.poly);
    const bool right_poly_ok = has_right && IsPolyUsable(lanes.right.poly);
    const float half_lane_m = cfg.lane_width_m * 0.5f;

    for (const float xq : sample_xs) {
        float y_center = 0.0f;

        if (has_left && has_right) {
            float yL = 0.0f, yR = 0.0f;
            const bool okL = SampleLaneYWithinKeypointRange(lanes.left, left_poly_ok, xq, yL);
            const bool okR = SampleLaneYWithinKeypointRange(lanes.right, right_poly_ok, xq, yR);
            if (!okL || !okR) continue;
            y_center = 0.5f * (yL + yR);
        } else if (has_left) {
            float yL = 0.0f;
            if (!SampleLaneYWithinKeypointRange(lanes.left, left_poly_ok, xq, yL)) continue;
            y_center = yL - half_lane_m;
        } else { // has_right
            float yR = 0.0f;
            if (!SampleLaneYWithinKeypointRange(lanes.right, right_poly_ok, xq, yR)) continue;
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

    // 4. 更新 Debug 字串
    std::ostringstream oss;
    oss << "A: has_L=" << has_left << " has_R=" << has_right
        << " | pts=" << out_centerline_box.kpts.size()
        << " | keypoint_xs=" << sample_xs.size();
    if (has_left) oss << " | L_fit{" << lanes.left.debug_info << "}";
    if (has_right) oss << " | R_fit{" << lanes.right.debug_info << "}";

    debug = oss.str();
    return true;
}

} // namespace internal
} // namespace lane_keeping
