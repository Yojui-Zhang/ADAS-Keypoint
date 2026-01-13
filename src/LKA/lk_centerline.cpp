#include "lk_centerline.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "lk_lane_points.h" // SampleYLinear
#include "lk_polyfit.h"     // PolyY
#include "lk_lane_selector.h" // [NEW] 引入選擇器

namespace lane_keeping {
namespace internal {

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

    // 3. 採樣生成中心線
    const int kNumSamples = 15;
    std::vector<cv::Point3f> center_kpts;
    center_kpts.reserve(kNumSamples);
    
    // 檢查多項式是否都有效 (有些情況可能只有點雲沒有擬合成功，雖然在 FindBestLaneCandidates 裡優先選有擬合的)
    // 這裡為了簡化，我們假設只要 valid 為 true 且 pts 不為空，就依賴 selector 的結果。
    // 但因為 selector 允許 "poly_ok = false" 的情況下選中車道 (fallback)，我們需檢查 poly 是否為零向量來決定是否用 PolyY
    // 為了代碼健壯性，我們簡單判斷 poly[0] != 0 (或其他檢查方式)，這裡沿用原本邏輯概念。
    
    // 簡單 helper lambda
    auto is_poly_valid = [](const cv::Vec3d& p) { return p != cv::Vec3d(0,0,0); };
    const bool left_poly_ok = has_left && is_poly_valid(lanes.left.poly);
    const bool right_poly_ok = has_right && is_poly_valid(lanes.right.poly);
    const float half_lane_m = cfg.lane_width_m * 0.5f;

    for (int i = 0; i < kNumSamples; ++i) {
        const float t = (kNumSamples == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(kNumSamples - 1);
        const float xq = x_min + t * (x_max - x_min);

        float y_center = 0.0f;

        if (has_left && has_right) {
            if (left_poly_ok && right_poly_ok) {
                const float yL = static_cast<float>(PolyY(lanes.left.poly,  xq));
                const float yR = static_cast<float>(PolyY(lanes.right.poly, xq));
                y_center = 0.5f * (yL + yR);
            } else {
                // Fallback: 線性採樣
                float yL = 0.0f, yR = 0.0f;
                bool okL = SampleYLinear(lanes.left.pts,  xq, yL);
                bool okR = SampleYLinear(lanes.right.pts, xq, yR);
                if (!okL || !okR) continue;
                y_center = 0.5f * (yL + yR);
            }
        } else if (has_left) {
            if (left_poly_ok) {
                y_center = static_cast<float>(PolyY(lanes.left.poly, xq)) - half_lane_m;
            } else {
                float yL = 0.0f;
                if (!SampleYLinear(lanes.left.pts, xq, yL)) continue;
                y_center = yL - half_lane_m;
            }
        } else { // has_right
            if (right_poly_ok) {
                y_center = static_cast<float>(PolyY(lanes.right.poly, xq)) + half_lane_m;
            } else {
                float yR = 0.0f;
                if (!SampleYLinear(lanes.right.pts, xq, yR)) continue;
                y_center = yR + half_lane_m;
            }
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
        << " | pts=" << out_centerline_box.kpts.size();
    if (has_left) oss << " | L_fit{" << lanes.left.debug_info << "}";
    if (has_right) oss << " | R_fit{" << lanes.right.debug_info << "}";

    debug = oss.str();
    return true;
}

} // namespace internal
} // namespace lane_keeping