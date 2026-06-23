#include "WorldGridOverlay.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "CameraModel.h"
#include "CameraProjectionUtils.h"
#include "config.h"

namespace {

void DrawOutlinedText(cv::Mat& image,
                      const std::string& text,
                      const cv::Point& origin,
                      const cv::Scalar& color,
                      double scale = 0.45,
                      int thickness = 1) {
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 2, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

bool DrawPolylineFromSampler(cv::Mat& image,
                             const std::function<cv::Point2f(float)>& sampler,
                             float start,
                             float end,
                             float step,
                             const cv::Scalar& color,
                             int thickness,
                             cv::Point* label_anchor,
                             bool draw_lines) {
    if (step <= 1e-6f || end < start) {
        return false;
    }

    const float padded_margin = 64.0f;
    std::vector<cv::Point> segment;
    bool have_label_anchor = false;

    for (float value = start; value <= end + step * 0.5f; value += step) {
        const cv::Point2f projected = sampler(value);
        const bool padded_visible =
            IsProjectedPointInsideImage(image.size(), projected, padded_margin);
        if (padded_visible == false) {
            if (draw_lines && segment.size() >= 2) {
                cv::polylines(image, segment, false, color, thickness, cv::LINE_AA);
            }
            segment.clear();
            continue;
        }

        const cv::Point rounded(cvRound(projected.x), cvRound(projected.y));
        segment.push_back(rounded);

        if (have_label_anchor == false &&
            IsProjectedPointInsideImage(image.size(), projected, 0.0f)) {
            if (label_anchor != nullptr) {
                *label_anchor = rounded;
            }
            have_label_anchor = true;
        }
    }

    if (draw_lines && segment.size() >= 2) {
        cv::polylines(image, segment, false, color, thickness, cv::LINE_AA);
    }

    return have_label_anchor;
}

bool IsMajorIndex(int index, int major_every_n) {
    if (index == 0) {
        return true;
    }
    if (major_every_n <= 0) {
        return false;
    }
    return (std::abs(index) % major_every_n) == 0;
}

std::string FormatMeters(float meters) {
    std::ostringstream oss;
    const float rounded = std::round(meters);
    if (std::fabs(meters - rounded) < 1e-3f) {
        oss << static_cast<int>(rounded);
    } else {
        oss << std::fixed << std::setprecision(1) << meters;
    }
    return oss.str();
}

std::string FormatForwardLabel(float forward_m) {
    std::ostringstream oss;
    oss << "F " << FormatMeters(forward_m) << "m";
    return oss.str();
}

std::string FormatLateralLabel(float left_m) {
    std::ostringstream oss;
    if (std::fabs(left_m) < 1e-4f) {
        oss << "L 0m";
    } else if (left_m > 0.0f) {
        oss << "L +" << FormatMeters(left_m) << "m";
    } else {
        oss << "L " << FormatMeters(left_m) << "m";
    }
    return oss.str();
}

void AppendPolylineCommandsFromSampler(adas_render::DrawCommandBuffer& commands,
                                       const cv::Size& image_size,
                                       const std::function<cv::Point2f(float)>& sampler,
                                       float start,
                                       float end,
                                       float step,
                                       const cv::Scalar& color,
                                       float thickness) {
    if (step <= 1e-6f || end < start) {
        return;
    }

    const float padded_margin = 64.0f;
    std::vector<cv::Point2f> segment;

    for (float value = start; value <= end + step * 0.5f; value += step) {
        const cv::Point2f projected = sampler(value);
        if (IsProjectedPointInsideImage(image_size, projected, padded_margin) == false) {
            for (size_t i = 1; i < segment.size(); ++i) {
                commands.AddLine(segment[i - 1], segment[i], color, thickness);
            }
            segment.clear();
            continue;
        }

        segment.push_back(projected);
    }

    for (size_t i = 1; i < segment.size(); ++i) {
        commands.AddLine(segment[i - 1], segment[i], color, thickness);
    }
}

}  // namespace

namespace {

void DrawWorldGridOverlayInternal(cv::Mat& image,
                                  const CameraModel& cam,
                                  const WorldGridOverlayConfig& cfg,
                                  bool draw_lines) {
    if (image.empty() || cfg.enabled == false) {
        return;
    }

    const float spacing_m = std::max(0.1f, cfg.spacing_m);
    const float sample_step_m = std::max(0.05f, cfg.sample_step_m);
    const float forward_start_m = std::max(0.1f, cfg.forward_start_m);
    const float forward_end_m = std::max(forward_start_m, cfg.forward_end_m);
    const float lateral_min_m = std::min(cfg.lateral_min_m, cfg.lateral_max_m);
    const float lateral_max_m = std::max(cfg.lateral_min_m, cfg.lateral_max_m);

    const int forward_begin_idx = static_cast<int>(std::ceil(forward_start_m / spacing_m));
    const int forward_end_idx = static_cast<int>(std::floor(forward_end_m / spacing_m));
    const int lateral_begin_idx = static_cast<int>(std::ceil(lateral_min_m / spacing_m));
    const int lateral_end_idx = static_cast<int>(std::floor(lateral_max_m / spacing_m));

    for (int idx = forward_begin_idx; idx <= forward_end_idx; ++idx) {
        const float forward_m = static_cast<float>(idx) * spacing_m;
        const bool major = IsMajorIndex(idx, cfg.major_every_n);
        const cv::Scalar color = major ? ORANGE : cv::Scalar(90, 90, 90);
        const int thickness = major ? 2 : 1;

        cv::Point label_anchor;
        const bool has_label_anchor = DrawPolylineFromSampler(
            image,
            [&](float left_m) {
                return ProjectVehicleGroundPointToImage(cam, image.size(), forward_m, left_m);
            },
            lateral_min_m,
            lateral_max_m,
            sample_step_m,
            color,
            thickness,
            &label_anchor,
            draw_lines);

        if (cfg.draw_labels && major && has_label_anchor) {
            DrawOutlinedText(image,
                             FormatForwardLabel(forward_m),
                             label_anchor + cv::Point(6, -6),
                             color);
        }
    }

    for (int idx = lateral_begin_idx; idx <= lateral_end_idx; ++idx) {
        const float left_m = static_cast<float>(idx) * spacing_m;
        const bool center_line = std::fabs(left_m) < 1e-4f;
        const bool major = IsMajorIndex(idx, cfg.major_every_n);
        const cv::Scalar color = center_line ? YELLOW : (major ? CYAN : cv::Scalar(110, 110, 110));
        const int thickness = center_line ? 2 : (major ? 2 : 1);

        cv::Point label_anchor;
        const bool has_label_anchor = DrawPolylineFromSampler(
            image,
            [&](float forward_m) {
                return ProjectVehicleGroundPointToImage(cam, image.size(), forward_m, left_m);
            },
            forward_start_m,
            forward_end_m,
            sample_step_m,
            color,
            thickness,
            &label_anchor,
            draw_lines);

        if (cfg.draw_labels && (center_line || major) && has_label_anchor) {
            DrawOutlinedText(image,
                             FormatLateralLabel(left_m),
                             label_anchor + cv::Point(6, 16),
                             color);
        }
    }
}

}  // namespace

void DrawWorldGridOverlay(cv::Mat& image,
                          const CameraModel& cam,
                          const WorldGridOverlayConfig& cfg) {
    DrawWorldGridOverlayInternal(image, cam, cfg, true);
}

void DrawWorldGridOverlayLabels(cv::Mat& image,
                                const CameraModel& cam,
                                const WorldGridOverlayConfig& cfg) {
    DrawWorldGridOverlayInternal(image, cam, cfg, false);
}

void AppendWorldGridOverlayCommands(adas_render::DrawCommandBuffer& commands,
                                    const cv::Size& image_size,
                                    const CameraModel& cam,
                                    const WorldGridOverlayConfig& cfg) {
    if (image_size.empty() || cfg.enabled == false) {
        return;
    }

    const float spacing_m = std::max(0.1f, cfg.spacing_m);
    const float sample_step_m = std::max(0.05f, cfg.sample_step_m);
    const float forward_start_m = std::max(0.1f, cfg.forward_start_m);
    const float forward_end_m = std::max(forward_start_m, cfg.forward_end_m);
    const float lateral_min_m = std::min(cfg.lateral_min_m, cfg.lateral_max_m);
    const float lateral_max_m = std::max(cfg.lateral_min_m, cfg.lateral_max_m);

    const int forward_begin_idx = static_cast<int>(std::ceil(forward_start_m / spacing_m));
    const int forward_end_idx = static_cast<int>(std::floor(forward_end_m / spacing_m));
    const int lateral_begin_idx = static_cast<int>(std::ceil(lateral_min_m / spacing_m));
    const int lateral_end_idx = static_cast<int>(std::floor(lateral_max_m / spacing_m));

    for (int idx = forward_begin_idx; idx <= forward_end_idx; ++idx) {
        const float forward_m = static_cast<float>(idx) * spacing_m;
        const bool major = IsMajorIndex(idx, cfg.major_every_n);
        const cv::Scalar color = major ? ORANGE : cv::Scalar(90, 90, 90);
        const float thickness = major ? 2.0f : 1.0f;

        AppendPolylineCommandsFromSampler(
            commands,
            image_size,
            [&](float left_m) {
                return ProjectVehicleGroundPointToImage(cam, image_size, forward_m, left_m);
            },
            lateral_min_m,
            lateral_max_m,
            sample_step_m,
            color,
            thickness);
    }

    for (int idx = lateral_begin_idx; idx <= lateral_end_idx; ++idx) {
        const float left_m = static_cast<float>(idx) * spacing_m;
        const bool center_line = std::fabs(left_m) < 1e-4f;
        const bool major = IsMajorIndex(idx, cfg.major_every_n);
        const cv::Scalar color = center_line ? YELLOW : (major ? CYAN : cv::Scalar(110, 110, 110));
        const float thickness = center_line ? 2.0f : (major ? 2.0f : 1.0f);

        AppendPolylineCommandsFromSampler(
            commands,
            image_size,
            [&](float forward_m) {
                return ProjectVehicleGroundPointToImage(cam, image_size, forward_m, left_m);
            },
            forward_start_m,
            forward_end_m,
            sample_step_m,
            color,
            thickness);
    }
}
