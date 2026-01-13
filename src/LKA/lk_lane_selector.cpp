#include "lk_lane_selector.h"

#include <algorithm>
#include <cmath>
#include "lk_lane_points.h"
#include "lk_polyfit.h"

namespace lane_keeping {
namespace internal {

LanePair FindBestLaneCandidates(const std::vector<TrackingBox>& world_result,
                                const ControlConfig& cfg)
{
    LanePair result;
    
    // 定義評估點 x (保持與原邏輯一致)
    const float x_eval = std::max(0.5f, std::min(cfg.x_heading_straight_m, 3.0f));

    for (const auto& box : world_result) {
        if (box.class_id != 0) continue;

        std::vector<cv::Point2f> pts;
        std::string one_debug;
        const LanePointStatus status = ExtractLanePointsVehicleM(box, cfg, pts, &one_debug);
        if (status != LanePointStatus::kOk) {
            continue;
        }

        // 嘗試擬合
        cv::Vec3d poly(0.0, 0.0, 0.0);
        std::string fit_dbg;
        const bool poly_ok = FitQuadraticLeastSquares(pts, poly, fit_dbg);

        // 計算評估用的 y 值
        float y_eval = 0.0f;
        if (poly_ok) {
            // 避免外插過遠
            const double xq = std::max<double>(pts.front().x,
                            std::min<double>(pts.back().x, x_eval));
            y_eval = static_cast<float>(PolyY(poly, xq));
        } else {
            // 如果擬合失敗，嘗試線性估計 (Fallback)
            if (!EstimateLaneYAtX(pts, x_eval, y_eval)) {
                continue;
            }
        }

        const float abs_y = std::fabs(y_eval);

        // 根據 y 值正負歸類為左或右，並保留 abs_y 最小 (最靠近預期位置) 的那一條
        if (y_eval > 0.0f) { // Left
            if (!result.left.valid || abs_y < result.left.abs_y_eval) {
                result.left.valid = true;
                result.left.abs_y_eval = abs_y;
                result.left.pts = std::move(pts); // 移動語意，減少拷貝
                if (poly_ok) {
                    result.left.poly = poly;
                    result.left.debug_info = fit_dbg;
                }
            }
        } else if (y_eval < 0.0f) { // Right
            if (!result.right.valid || abs_y < result.right.abs_y_eval) {
                result.right.valid = true;
                result.right.abs_y_eval = abs_y;
                result.right.pts = std::move(pts);
                if (poly_ok) {
                    result.right.poly = poly;
                    result.right.debug_info = fit_dbg;
                }
            }
        }
    }
    return result;
}

} // namespace internal
} // namespace lane_keeping
