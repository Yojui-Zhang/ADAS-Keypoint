#include "lk_visualization.h"

#include <vector>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <functional>
#include <string>
#include <opencv2/imgproc.hpp>

#include "lk_centerline.h"
#include "lk_lane_points.h"
#include "lk_polyfit.h"
#include "lk_lane_selector.h" // [NEW] 引入選擇器
#include "CameraProjectionUtils.h"
#include "config.h"

namespace lane_keeping {
namespace internal {

namespace {

constexpr std::size_t kMaxVisualKeypointSamples = 15;

struct DirectLaneCandidate {
    bool valid = false;
    std::vector<cv::Point2f> pts;
    float abs_y_eval = 1e9f;
};

struct DirectLanePair {
    DirectLaneCandidate left;
    DirectLaneCandidate right;
};

struct LaneDetectLaneModel {
    bool valid = false;
    std::vector<cv::Point2f> pts;
    cv::Vec3d poly = {0.0, 0.0, 0.0};
    bool use_poly = false;
};

struct LaneDetectPair {
    LaneDetectLaneModel left;
    LaneDetectLaneModel right;
};

void DrawOutlinedText(cv::Mat& output_img,
                      const std::string& text,
                      const cv::Point& origin,
                      const cv::Scalar& color) {
    cv::putText(output_img, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 3, cv::LINE_AA);
    cv::putText(output_img, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
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
                                         float x_start_m,
                                         float x_end_m) {
    const float lo = std::min(x_start_m, x_end_m);
    const float hi = std::max(x_start_m, x_end_m);
    std::vector<float> xs;
    xs.reserve(std::min<std::size_t>(pts.size(), kMaxVisualKeypointSamples));

    for (const auto& p : pts) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        if (p.x < lo || p.x > hi) {
            continue;
        }
        xs.push_back(p.x);
    }

    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end(), [](float a, float b) {
                 return std::fabs(a - b) <= 1e-3f;
             }),
             xs.end());
    return DownsampleXsToMax(xs, kMaxVisualKeypointSamples);
}

bool DrawProjectedKeypointPolyline(cv::Mat& output_img,
                                   const CameraModel& cam,
                                   const std::vector<float>& x_samples,
                                   const std::function<bool(float, float&)>& sampler,
                                   const cv::Scalar& color,
                                   int thickness,
                                   const std::string& label) {
    if (output_img.empty() || x_samples.size() < 2) {
        return false;
    }

    std::vector<cv::Point> draw_pts;
    draw_pts.reserve(x_samples.size());
    cv::Point label_pt;
    bool label_valid = false;

    for (const float xq : x_samples) {
        float yq = 0.0f;
        if (!sampler(xq, yq)) {
            continue;
        }

        const cv::Point2f uv = ProjectVehicleGroundPointToImage(cam, output_img.size(), xq, yq);
        if (!IsProjectedPointInsideImage(output_img.size(), uv, 0.0f)) {
            continue;
        }

        const cv::Point rounded(cvRound(uv.x), cvRound(uv.y));
        draw_pts.push_back(rounded);
        if (!label_valid) {
            label_pt = rounded;
            label_valid = true;
        }
    }

    if (draw_pts.size() < 2) {
        return false;
    }

    cv::polylines(output_img, draw_pts, false, color, thickness, cv::LINE_AA);
    if (label_valid && !label.empty()) {
        DrawOutlinedText(output_img, label, label_pt + cv::Point(8, -6), color);
    }
    return true;
}

DirectLanePair SelectDirectLaneCandidates(const std::vector<TrackingBox>& world_result,
                                          const ControlConfig& cfg) {
    DirectLanePair result;
    const float x_eval = std::max(0.5f, std::min(cfg.x_heading_straight_m,
                                                 std::max(0.5f, cfg.lane_detect_forward_range_m)));

    for (const auto& box : world_result) {
        std::vector<cv::Point2f> pts;
        if (ExtractLanePointsVehicleM(box, cfg, pts, nullptr) != LanePointStatus::kOk) {
            continue;
        }

        float y_eval = 0.0f;
        if (!EstimateLaneYAtX(pts, x_eval, y_eval)) {
            continue;
        }

        const float abs_y = std::fabs(y_eval);
        if (y_eval > 0.0f) {
            if (!result.left.valid || abs_y < result.left.abs_y_eval) {
                result.left.valid = true;
                result.left.abs_y_eval = abs_y;
                result.left.pts = std::move(pts);
            }
        } else if (y_eval < 0.0f) {
            if (!result.right.valid || abs_y < result.right.abs_y_eval) {
                result.right.valid = true;
                result.right.abs_y_eval = abs_y;
                result.right.pts = std::move(pts);
            }
        }
    }

    return result;
}

std::string ToLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool IsLanePolyUsable(const cv::Vec3d& poly) {
    return poly != cv::Vec3d(0.0, 0.0, 0.0);
}

LaneDetectPair BuildLaneDetectPair(const std::vector<TrackingBox>& world_result,
                                   const ControlConfig& cfg) {
    LaneDetectPair result;
    const std::string mode = ToLowerCopy(cfg.lane_detect_mode);

    if (mode == "quadratic_curve" || mode == "quadratic" || mode == "poly") {
        const LanePair lanes = FindBestLaneCandidates(world_result, cfg);
        result.left.valid = lanes.left.valid;
        result.left.pts = lanes.left.pts;
        result.left.poly = lanes.left.poly;
        result.left.use_poly = lanes.left.valid && IsLanePolyUsable(lanes.left.poly);

        result.right.valid = lanes.right.valid;
        result.right.pts = lanes.right.pts;
        result.right.poly = lanes.right.poly;
        result.right.use_poly = lanes.right.valid && IsLanePolyUsable(lanes.right.poly);
        return result;
    }

    const DirectLanePair direct = SelectDirectLaneCandidates(world_result, cfg);
    result.left.valid = direct.left.valid;
    result.left.pts = direct.left.pts;
    result.right.valid = direct.right.valid;
    result.right.pts = direct.right.pts;
    return result;
}

bool SampleLaneModelY(const LaneDetectLaneModel& lane,
                      float xq,
                      float& yq) {
    if (!lane.valid || lane.pts.size() < 2) {
        return false;
    }

    if (xq < lane.pts.front().x || xq > lane.pts.back().x) {
        return false;
    }

    if (lane.use_poly) {
        yq = static_cast<float>(PolyY(lane.poly, static_cast<double>(xq)));
        return true;
    }

    return SampleYLinear(lane.pts, xq, yq);
}

bool DrawLaneModel(cv::Mat& output_img,
                   const CameraModel& cam,
                   const LaneDetectLaneModel& lane,
                   float x_start_m,
                   float x_end_m,
                   const cv::Scalar& color,
                   int thickness,
                   const std::string& label,
                   bool label_near_bottom) {
    if (!lane.valid || lane.pts.size() < 2 || x_end_m <= x_start_m) {
        return false;
    }

    const float sample_start = std::max(x_start_m, lane.pts.front().x);
    const float sample_end = std::min(x_end_m, lane.pts.back().x);
    if (sample_end <= sample_start) {
        return false;
    }

    constexpr int kSampleCount = 80;
    std::vector<cv::Point> draw_pts;
    draw_pts.reserve(kSampleCount);
    cv::Point label_pt;
    bool label_valid = false;

    for (int i = 0; i < kSampleCount; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSampleCount - 1);
        const float xq = sample_start + t * (sample_end - sample_start);
        float yq = 0.0f;
        if (!SampleLaneModelY(lane, xq, yq)) {
            continue;
        }

        const cv::Point2f uv = ProjectVehicleGroundPointToImage(cam, output_img.size(), xq, yq);
        if (!IsProjectedPointInsideImage(output_img.size(), uv, 0.0f)) {
            continue;
        }

        const cv::Point rounded(cvRound(uv.x), cvRound(uv.y));
        draw_pts.push_back(rounded);
        if (!label_valid ||
            (label_near_bottom && rounded.y > label_pt.y) ||
            (!label_near_bottom && rounded.y < label_pt.y)) {
            label_pt = rounded;
            label_valid = true;
        }
    }

    if (draw_pts.size() < 2) {
        return false;
    }

    cv::polylines(output_img, draw_pts, false, color, thickness, cv::LINE_AA);
    if (label_valid && !label.empty()) {
        const cv::Point offset = label_near_bottom ? cv::Point(8, 18) : cv::Point(8, -6);
        DrawOutlinedText(output_img, label, label_pt + offset, color);
    }
    return true;
}

bool LaneTouchesVehicleEdge(const LaneDetectLaneModel& lane,
                            float vehicle_half_width_m,
                            float forward_range_m,
                            float contact_margin_m,
                            bool is_left_lane) {
    if (!lane.valid || lane.pts.size() < 2) {
        return false;
    }

    const float sample_start = std::max(0.0f, lane.pts.front().x);
    const float sample_end = std::min(std::max(0.5f, forward_range_m), lane.pts.back().x);
    if (sample_end <= sample_start) {
        return false;
    }

    constexpr int kSampleCount = 30;
    for (int i = 0; i < kSampleCount; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSampleCount - 1);
        const float xq = sample_start + t * (sample_end - sample_start);
        float yq = 0.0f;
        if (!SampleLaneModelY(lane, xq, yq)) {
            continue;
        }

        if (is_left_lane) {
            if (yq <= vehicle_half_width_m + contact_margin_m) {
                return true;
            }
        } else {
            if (yq >= -(vehicle_half_width_m + contact_margin_m)) {
                return true;
            }
        }
    }

    return false;
}

bool RawKeypointsTouchVehicleEdge(const std::vector<cv::Point2f>& pts,
                                  float vehicle_half_width_m,
                                  float forward_range_m,
                                  float contact_margin_m,
                                  bool is_left_lane) {
    if (pts.empty()) {
        return false;
    }

    const float x_end = std::max(0.5f, forward_range_m);
    const float edge_threshold = vehicle_half_width_m + contact_margin_m;

    for (const auto& p : pts) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        if (p.x < 0.0f || p.x > x_end) {
            continue;
        }

        if (is_left_lane) {
            if (p.y <= edge_threshold) {
                return true;
            }
        } else {
            if (p.y >= -edge_threshold) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace

LaneDepartureStatus DetectRawLaneDepartureFromKeypoints(
    const std::vector<TrackingBox>& world_result,
    const ControlConfig& cfg) {
    LaneDepartureStatus status;

    const DirectLanePair lanes = SelectDirectLaneCandidates(world_result, cfg);
    const float vehicle_half_width_m = std::max(0.1f, cfg.lane_detect_vehicle_half_width_m);
    const float detect_range_m = std::max(0.5f, cfg.lane_detect_forward_range_m);

    status.left_departure = lanes.left.valid &&
        RawKeypointsTouchVehicleEdge(lanes.left.pts,
                                     vehicle_half_width_m,
                                     detect_range_m,
                                     cfg.lane_detect_contact_margin_m,
                                     true);
    status.right_departure = lanes.right.valid &&
        RawKeypointsTouchVehicleEdge(lanes.right.pts,
                                     vehicle_half_width_m,
                                     detect_range_m,
                                     cfg.lane_detect_contact_margin_m,
                                     false);
    status.departure = status.left_departure || status.right_departure;
    return status;
}

void DrawCenterlineOnImage(const TrackingBox& center_lane,
                           cv::Mat& output_img,
                           const CameraModel& cam) {
    if (output_img.empty()) return;
    std::vector<cv::Point> draw_pts;
    draw_pts.reserve(center_lane.kpts.size());
    for (const auto& kp : center_lane.kpts) {
        const cv::Point2f uv = ProjectVehicleGroundPointToImage(cam, output_img.size(), kp.x, kp.y);
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

    auto draw_lane = [&](const LaneCandidate& lane,
                         const cv::Scalar& color,
                         int thickness) {
        if (!lane.valid || lane.pts.size() < 2) return;

        const bool poly_ok = lane.poly != cv::Vec3d(0.0, 0.0, 0.0);
        const std::vector<float> xs = CollectPointXsInRange(lane.pts, x_min, x_max);
        DrawProjectedKeypointPolyline(
            output_img,
            cam,
            xs,
            [&](float xq, float& yq) {
                if (xq < lane.pts.front().x || xq > lane.pts.back().x) {
                    return false;
                }
                if (poly_ok) {
                    yq = static_cast<float>(PolyY(lane.poly, static_cast<double>(xq)));
                    return true;
                }
                return SampleYLinear(lane.pts, xq, yq);
            },
            color,
            thickness,
            "");
    };

    // Left: Green, Right: Blue
    draw_lane(lanes.left, cv::Scalar(0, 255, 0), 2);
    draw_lane(lanes.right, cv::Scalar(255, 0, 0), 2);
}

void DrawLkaLaneSolutionOnImage(const std::vector<TrackingBox>& world_result,
                                cv::Mat& output_img,
                                const CameraModel& cam,
                                const ControlConfig& cfg,
                                float x_start_m,
                                float x_end_m) {
    if (output_img.empty()) {
        return;
    }

    const float draw_x_start_m = std::max(0.0f, std::min(x_start_m, x_end_m));
    const float draw_x_end_m = std::max(draw_x_start_m, std::max(x_start_m, x_end_m));

    const LanePair lanes = FindBestLaneCandidates(world_result, cfg);

    auto draw_lane_candidate = [&](const LaneCandidate& lane,
                                   float lateral_offset_m,
                                   const cv::Scalar& color,
                                   const std::string& label) {
        if (!lane.valid) {
            return false;
        }

        const bool poly_ok = lane.poly != cv::Vec3d(0.0, 0.0, 0.0);
        const std::vector<float> xs =
            CollectPointXsInRange(lane.pts, draw_x_start_m, draw_x_end_m);

        return DrawProjectedKeypointPolyline(
            output_img,
            cam,
            xs,
            [&](float xq, float& yq) {
                if (xq < lane.pts.front().x || xq > lane.pts.back().x) {
                    return false;
                }
                if (poly_ok) {
                    yq = static_cast<float>(PolyY(lane.poly, static_cast<double>(xq))) + lateral_offset_m;
                    return true;
                }
                const bool ok = SampleYLinear(lane.pts, xq, yq);
                yq += lateral_offset_m;
                return ok;
            },
            color,
            2,
            label);
    };

    draw_lane_candidate(lanes.left, 0.0f, cv::Scalar(0, 255, 0), "L lane");
    draw_lane_candidate(lanes.right, 0.0f, cv::Scalar(255, 0, 0), "R lane");

    TrackingBox center_lane;
    std::string centerline_debug;
    if (!BuildCenterlineFromWorldResult(world_result, cfg, center_lane, centerline_debug)) {
        return;
    }
    (void)centerline_debug;

    std::vector<cv::Point2f> center_pts;
    center_pts.reserve(center_lane.kpts.size());
    for (const auto& kp : center_lane.kpts) {
        if (!std::isfinite(kp.x) || !std::isfinite(kp.y)) {
            continue;
        }
        center_pts.emplace_back(kp.x, kp.y);
    }
    if (center_pts.size() < 2) {
        return;
    }

    cv::Vec3d center_poly(0.0, 0.0, 0.0);
    std::string center_fit_debug;
    const bool center_poly_ok = FitQuadraticLeastSquares(center_pts, center_poly, center_fit_debug);
    (void)center_fit_debug;

    const std::vector<float> center_xs =
        CollectPointXsInRange(center_pts, draw_x_start_m, draw_x_end_m);
    DrawProjectedKeypointPolyline(
        output_img,
        cam,
        center_xs,
        [&](float xq, float& yq) {
            const float center_x_min = center_pts.front().x;
            const float center_x_max = center_pts.back().x;
            if (xq < center_x_min || xq > center_x_max) {
                return false;
            }
            if (center_poly_ok) {
                yq = static_cast<float>(PolyY(center_poly, static_cast<double>(xq)));
                return true;
            }
            return SampleYLinear(center_pts, xq, yq);
        },
        cv::Scalar(0, 165, 255),
        2,
        "Center");
}

void DrawLaneDetectOverlayOnImage(const std::vector<TrackingBox>& world_result,
                                  cv::Mat& output_img,
                                  const CameraModel& cam,
                                  const ControlConfig& cfg) {
    if (output_img.empty()) {
        return;
    }

    const LaneDetectPair lanes = BuildLaneDetectPair(world_result, cfg);
    const float draw_end_m = std::max(0.5f, cfg.lane_detect_draw_end_m);
    const float bottom_range_m = std::min(draw_end_m, std::max(0.5f, cfg.lane_detect_bottom_range_m));
    const float vehicle_half_width_m = std::max(0.1f, cfg.lane_detect_vehicle_half_width_m);
    const float detect_range_m = std::max(0.5f, cfg.lane_detect_forward_range_m);

    const bool left_departure = lanes.left.valid &&
        LaneTouchesVehicleEdge(lanes.left,
                               vehicle_half_width_m,
                               detect_range_m,
                               cfg.lane_detect_contact_margin_m,
                               true);
    const bool right_departure = lanes.right.valid &&
        LaneTouchesVehicleEdge(lanes.right,
                               vehicle_half_width_m,
                               detect_range_m,
                               cfg.lane_detect_contact_margin_m,
                               false);

    if (lanes.left.valid) {
        const cv::Scalar lane_color = left_departure ? RED : GREEN;
        DrawLaneModel(output_img,
                      cam,
                      lanes.left,
                      0.0f,
                      draw_end_m,
                      lane_color,
                      left_departure ? 3 : 2,
                      left_departure ? "LaneDet L" : "",
                      false);
        if (left_departure) {
            DrawLaneModel(output_img,
                          cam,
                          lanes.left,
                          0.0f,
                          bottom_range_m,
                          RED,
                          5,
                          "Depart L",
                          true);
        }
    }

    if (lanes.right.valid) {
        const cv::Scalar lane_color = right_departure ? RED : GREEN;
        DrawLaneModel(output_img,
                      cam,
                      lanes.right,
                      0.0f,
                      draw_end_m,
                      lane_color,
                      right_departure ? 3 : 2,
                      right_departure ? "LaneDet R" : "",
                      false);
        if (right_departure) {
            DrawLaneModel(output_img,
                          cam,
                          lanes.right,
                          0.0f,
                          bottom_range_m,
                          RED,
                          5,
                          "Depart R",
                          true);
        }
    }

    if (left_departure || right_departure) {
        std::string status = "Lane detect: ";
        if (left_departure && right_departure) {
            status += "LEFT+RIGHT";
        } else if (left_departure) {
            status += "LEFT";
        } else {
            status += "RIGHT";
        }
        DrawOutlinedText(output_img,
                         status,
                         cv::Point(30, std::max(40, output_img.rows - 30)),
                         RED);
    }
}

} // namespace internal
} // namespace lane_keeping
