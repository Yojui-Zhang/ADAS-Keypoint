#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "config.h"

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
    int block_gap_y = 16;
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
                      int foreground_thickness) {
    cv::line(img, start, end, BLACK, foreground_thickness + 2, cv::LINE_AA);
    cv::line(img, start, end, WHITE, foreground_thickness, cv::LINE_AA);
}

void DrawCurrentTargetBlock(cv::Mat& img,
                            cv::Point current_origin,
                            const std::string& current_text,
                            const std::string& target_text,
                            const CurrentTargetBlockLayout& layout,
                            const TargetInfoStyle& style) {
    DrawOutlinedText(img, current_text, current_origin,
                     style.font_face, style.current_font_scale,
                     style.current_thickness, WHITE, style.outline_extra);

    DrawOutlinedLine(img,
                     current_origin + layout.slash_start,
                     current_origin + layout.slash_end,
                     style.current_thickness);

    DrawOutlinedText(img, target_text,
                     current_origin + cv::Point(layout.target_x, layout.target_baseline_y),
                     style.font_face, style.target_font_scale,
                     style.target_thickness, GRAY, style.outline_extra);
}

}  // namespace

void DrawTargetInfo(cv::Mat& img,
                    float current_speed_kmh,
                    float target_speed_kmh,
                    float current_steer_deg,
                    float target_steer_deg,
                    float current_brake_0_10,
                    float target_brake_0_10) {
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

    const std::string current_brake_text = FormatCurrentTarget(current_brake_0_10, "");
    const std::string target_brake_text = FormatCurrentTarget(target_brake_0_10, "");

    const CurrentTargetBlockLayout speed_layout =
        MeasureCurrentTargetBlock(current_speed_text, target_speed_text, style);
    const CurrentTargetBlockLayout steer_layout =
        MeasureCurrentTargetBlock(current_steer_text, target_steer_text, style);
    const CurrentTargetBlockLayout brake_layout =
        MeasureCurrentTargetBlock(current_brake_text, target_brake_text, style);

    const cv::Point speed_origin(left, top);
    const cv::Point steer_origin(left + speed_layout.width + style.block_gap_x, top);
    const int first_row_target_baseline_y =
        std::max(speed_layout.target_baseline_y, steer_layout.target_baseline_y);
    const int first_row_current_height =
        std::max(speed_layout.current_height, steer_layout.current_height);
    // const cv::Point brake_origin(
    //     left,
    //     top + first_row_target_baseline_y + first_row_current_height + style.block_gap_y);

    const cv::Point brake_origin(left + steer_layout.width + speed_layout.width + style.block_gap_x + style.block_gap_x, top);

    DrawCurrentTargetBlock(img, speed_origin, current_speed_text, target_speed_text,
                           speed_layout, style);
    DrawCurrentTargetBlock(img, steer_origin, current_steer_text, target_steer_text,
                           steer_layout, style);
    DrawCurrentTargetBlock(img, brake_origin, current_brake_text, target_brake_text,
                           brake_layout, style);

}
