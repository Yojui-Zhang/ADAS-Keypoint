#include "VehicleSkeletonDrawer.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace vehicle_skeleton {

static bool getPixelKpts(const TrackingBox& tb, std::vector<cv::Point2f>& out) {
    out.clear();

    const std::vector<cv::Point3f>* src = nullptr;
    if (!tb.track_kpt_history.empty()) {
        src = &tb.track_kpt_history.back();
    } else if (!tb.last_track_kpts.empty()) {
        src = &tb.last_track_kpts;
    } else {
        return false;
    }

    out.reserve(src->size());
    for (const auto& p : *src) out.emplace_back(p.x, p.y);
    return !out.empty();
}

static bool idxOk(const std::vector<cv::Point2f>& v, int idx) {
    return idx >= 0 && idx < static_cast<int>(v.size())
        && std::isfinite(v[idx].x) && std::isfinite(v[idx].y);
}

static void drawEdge(cv::Mat& img, const std::vector<cv::Point2f>& k, int a, int b,
                     const cv::Scalar& color, int thickness)
{
    if (!idxOk(k, a) || !idxOk(k, b)) return;
    cv::line(img, k[a], k[b], color, thickness, cv::LINE_AA);
}

static void drawRectLayer(cv::Mat& img, const std::vector<cv::Point2f>& k,
                          int lf, int rf, int lr, int rr,
                          const cv::Scalar& color, int thickness)
{
    drawEdge(img, k, lf, rf, color, thickness); // front
    drawEdge(img, k, rf, rr, color, thickness); // right
    drawEdge(img, k, rr, lr, color, thickness); // rear
    drawEdge(img, k, lr, lf, color, thickness); // left
}

bool DrawVehicleSkeletonOnImage(
    cv::Mat& img,
    const TrackingBox& tb,
    const SkeletonKptLayout& layout,
    const SkeletonDrawParams& draw_params
) {
    if (img.empty()) return false;

    std::vector<cv::Point2f> pix;
    if (!getPixelKpts(tb, pix)) return false;

    // 3 layers rectangles
    drawRectLayer(img, pix, layout.top_lf, layout.top_rf, layout.top_lr, layout.top_rr,
                  draw_params.color, draw_params.thickness);
    drawRectLayer(img, pix, layout.mid_lf, layout.mid_rf, layout.mid_lr, layout.mid_rr,
                  draw_params.color, draw_params.thickness);
    drawRectLayer(img, pix, layout.bot_lf, layout.bot_rf, layout.bot_lr, layout.bot_rr,
                  draw_params.color, draw_params.thickness);

    // vertical edges (top-mid-bot at each corner)
    drawEdge(img, pix, layout.top_lf, layout.mid_lf, draw_params.color, draw_params.thickness);
    drawEdge(img, pix, layout.mid_lf, layout.bot_lf, draw_params.color, draw_params.thickness);

    drawEdge(img, pix, layout.top_rf, layout.mid_rf, draw_params.color, draw_params.thickness);
    drawEdge(img, pix, layout.mid_rf, layout.bot_rf, draw_params.color, draw_params.thickness);

    drawEdge(img, pix, layout.top_lr, layout.mid_lr, draw_params.color, draw_params.thickness);
    drawEdge(img, pix, layout.mid_lr, layout.bot_lr, draw_params.color, draw_params.thickness);

    drawEdge(img, pix, layout.top_rr, layout.mid_rr, draw_params.color, draw_params.thickness);
    drawEdge(img, pix, layout.mid_rr, layout.bot_rr, draw_params.color, draw_params.thickness);

    // draw keypoints (optional)
    if (draw_params.draw_kpts) {
        for (int i = 0; i < static_cast<int>(pix.size()); ++i) {
            if (!std::isfinite(pix[i].x) || !std::isfinite(pix[i].y)) continue;
            cv::circle(img, pix[i], draw_params.kpt_radius, draw_params.color, -1, cv::LINE_AA);
        }
    }

    return true;
}

} // namespace vehicle_skeleton

