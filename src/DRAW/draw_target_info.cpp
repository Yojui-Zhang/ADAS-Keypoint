#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "draw_target_info.h"

namespace {

struct TextMetrics {
    cv::Size size;
    int baseline = 0;
};

struct TargetInfoStyle {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double current_font_scale = 1.5;
    double target_font_scale = 1.0;
    int current_thickness = 3;
    int target_thickness = 3;
    int outline_extra = 3;
    int slash_gap_x = 40;
    int slash_dx = -100;
    int slash_top_padding = 2;
    int text_row_gap = 0;
    int block_gap_x = 30;
    int brake_bar_bottom_margin = 28;
    int brake_bar_outer_thickness = 13;
    int brake_bar_inner_thickness = 7;
    int brake_bar_min_inner_px = 4;
    double ttc_font_scale = 1.0;
    int ttc_thickness = 3;
    int ttc_right_margin = 18;
    int ttc_bottom_margin = 28;
};

struct CurrentTargetBlockLayout {
    int width = 0;
    int current_height = 0;
    int target_baseline_y = 0;
    int target_x = 0;
    cv::Point slash_start;
    cv::Point slash_end;
};

std::string FormatCurrentTarget(float value, const std::string& label) {
    if (std::fabs(value) < 0.05f) {
        value = 0.0f;
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << value;
    std::string str = ss.str();

    if (str.find('.') != std::string::npos) {
        while (str.back() == '0') {
            str.pop_back();
        }
        if (str.back() == '.') {
            str.pop_back();
        }
    }

    return str + label;
}

std::string FormatTargetTtc(float ttc_s) {
    std::ostringstream ss;
    ss << "TTC ";

    if (!std::isfinite(ttc_s) || ttc_s <= 0.0f || ttc_s > 7.0f) {
        ss << "--";
        return ss.str();
    }

    ss << std::fixed << std::setprecision(1) << ttc_s << "s";
    return ss.str();
}

TextMetrics MeasureText(const std::string& text,
                        int font_face,
                        double font_scale,
                        int thickness) {
    TextMetrics metrics;
    metrics.size = cv::getTextSize(text, font_face, font_scale, thickness, &metrics.baseline);
    return metrics;
}

CurrentTargetBlockLayout MeasureCurrentTargetBlock(const std::string& current_text,
                                                   const std::string& target_text,
                                                   const TargetInfoStyle& style) {
    const TextMetrics current_metrics =
        MeasureText(current_text, style.font_face, style.current_font_scale, style.current_thickness);
    const TextMetrics target_metrics =
        MeasureText(target_text, style.font_face, style.target_font_scale, style.target_thickness);

    CurrentTargetBlockLayout layout;
    layout.current_height = current_metrics.size.height;
    layout.target_baseline_y =
        current_metrics.size.height + target_metrics.size.height + style.text_row_gap;
    layout.slash_start = cv::Point(current_metrics.size.width + style.slash_gap_x,
                                   -current_metrics.size.height - style.slash_top_padding);
    layout.slash_end = layout.slash_start +
                       cv::Point(style.slash_dx,
                                 layout.target_baseline_y - layout.slash_start.y);
    layout.target_x = std::max(0, layout.slash_end.x + style.slash_gap_x);
    layout.width = std::max(current_metrics.size.width,
                            std::max(layout.slash_start.x,
                                     std::max(layout.slash_end.x,
                                              layout.target_x + target_metrics.size.width)));
    return layout;
}

void DrawOutlinedText(cv::Mat& img,
                      const std::string& text,
                      cv::Point origin,
                      int font_face,
                      double font_scale,
                      int thickness,
                      const cv::Scalar& color,
                      int outline_extra) {
    cv::putText(img, text, origin, font_face, font_scale,
                BLACK, thickness + outline_extra, cv::LINE_AA);
    cv::putText(img, text, origin, font_face, font_scale,
                color, thickness, cv::LINE_AA);
}

void DrawOutlinedLine(cv::Mat& img,
                      cv::Point start,
                      cv::Point end,
                      int foreground_thickness,
                      const cv::Scalar& color) {
    cv::line(img, start, end, BLACK, foreground_thickness + 2, cv::LINE_AA);
    cv::line(img, start, end, color, foreground_thickness, cv::LINE_AA);
}

void DrawCurrentTargetBlock(cv::Mat& img,
                            cv::Point current_origin,
                            const std::string& current_text,
                            const std::string& target_text,
                            const CurrentTargetBlockLayout& layout,
                            const TargetInfoStyle& style,
                            const cv::Scalar& current_color,
                            const cv::Scalar& target_color,
                            const cv::Scalar& slash_color) {
    DrawOutlinedText(img, current_text, current_origin,
                     style.font_face, style.current_font_scale,
                     style.current_thickness, current_color, style.outline_extra);

    DrawOutlinedLine(img,
                     current_origin + layout.slash_start,
                     current_origin + layout.slash_end,
                     style.current_thickness,
                     slash_color);

    DrawOutlinedText(img, target_text,
                     current_origin + cv::Point(layout.target_x, layout.target_baseline_y),
                     style.font_face, style.target_font_scale,
                     style.target_thickness, target_color, style.outline_extra);
}

float ClampBrakeValue(float value) {
    if (std::isfinite(value) == false) {
        return 0.0f;
    }
    return std::clamp(value, 0.0f, 10.0f);
}

void DrawBrakeControlBar(cv::Mat& img,
                         int left,
                         int right_limit,
                         float brake_0_10,
                         bool active,
                         const TargetInfoStyle& style) {
    if (img.empty() || right_limit <= left) {
        return;
    }

    const int right = std::min(right_limit, img.cols - 8);
    const int width = std::max(1, right - left);
    const int y = std::max(0, img.rows - style.brake_bar_bottom_margin);
    const float brake = ClampBrakeValue(brake_0_10);
    const int fill_width = std::max(style.brake_bar_min_inner_px,
                                    static_cast<int>(std::round(width * (brake / 10.0f))));
    const int fill_right = std::clamp(left + fill_width, left, right);
    const cv::Scalar fill_color = active ? RED : WHITE;

    cv::line(img,
             cv::Point(left, y),
             cv::Point(right, y),
             BLACK,
             style.brake_bar_outer_thickness,
             cv::LINE_AA);

    if (fill_right > left) {
        cv::line(img,
                 cv::Point(left, y),
                 cv::Point(fill_right, y),
                 fill_color,
                 style.brake_bar_inner_thickness,
                 cv::LINE_AA);
    }
}

void DrawTargetTtc(cv::Mat& img,
                   float target_ttc_s,
                   const TargetInfoStyle& style) {
    const std::string ttc_text = FormatTargetTtc(target_ttc_s);
    const TextMetrics metrics =
        MeasureText(ttc_text, style.font_face, style.ttc_font_scale, style.ttc_thickness);
    const int x = std::max(style.ttc_right_margin,
                           img.cols - style.ttc_right_margin - metrics.size.width);
    const int y = std::max(metrics.size.height + metrics.baseline,
                           img.rows - style.ttc_bottom_margin);

    DrawOutlinedText(img,
                     ttc_text,
                     cv::Point(x, y),
                     style.font_face,
                     style.ttc_font_scale,
                     style.ttc_thickness,
                     WHITE,
                     style.outline_extra);
}

}  // namespace

void DrawTargetInfo(cv::Mat& img,
                    float current_speed_kmh,
                    float target_speed_kmh,
                    float current_steer_deg,
                    float target_steer_deg,
                    float current_brake_0_10,
                    float target_ttc_s,
                    const TargetInfoControlState& control_state) {
    if (img.empty()) {
        return;
    }

    const TargetInfoStyle style;
    const int left = 8;
    const int top = 50;

    const std::string current_speed_text = FormatCurrentTarget(current_speed_kmh, "");
    const std::string target_speed_text = FormatCurrentTarget(target_speed_kmh, " km/h");

    const std::string current_steer_text = FormatCurrentTarget(current_steer_deg, "");
    const std::string target_steer_text = FormatCurrentTarget(target_steer_deg, " deg");

    const CurrentTargetBlockLayout speed_layout =
        MeasureCurrentTargetBlock(current_speed_text, target_speed_text, style);
    const CurrentTargetBlockLayout steer_layout =
        MeasureCurrentTargetBlock(current_steer_text, target_steer_text, style);

    const cv::Point speed_origin(left, top);
    const cv::Point steer_origin(left + speed_layout.width + style.block_gap_x, top);
    const cv::Scalar speed_current_color = control_state.speed_control_active ? GREEN : WHITE;
    const cv::Scalar speed_target_color = control_state.speed_control_active ? GREEN : GRAY;
    const cv::Scalar speed_slash_color = control_state.speed_control_active ? GREEN : WHITE;
    const cv::Scalar steer_current_color = control_state.steering_control_active ? GREEN : WHITE;
    const cv::Scalar steer_target_color = control_state.steering_control_active ? GREEN : GRAY;
    const cv::Scalar steer_slash_color = control_state.steering_control_active ? GREEN : WHITE;

    DrawCurrentTargetBlock(img, speed_origin, current_speed_text, target_speed_text,
                           speed_layout, style,
                           speed_current_color, speed_target_color, speed_slash_color);
    DrawCurrentTargetBlock(img, steer_origin, current_steer_text, target_steer_text,
                           steer_layout, style,
                           steer_current_color, steer_target_color, steer_slash_color);

    DrawBrakeControlBar(img,
                        left,
                        // steer_right,
                        left + 300,
                        current_brake_0_10,
                        control_state.brake_control_active,
                        style);
    DrawTargetTtc(img, target_ttc_s, style);
}
