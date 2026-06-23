#include "adas_overlay.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include "config.h"

namespace {

void DrawOutlinedText(cv::Mat& image,
                      const std::string& text,
                      const cv::Point& origin,
                      const cv::Scalar& color) {
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 3, cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv::LINE_AA);
}

}  // namespace

namespace adas_render {

void AppendLkaReferenceOverlayCommands(DrawCommandBuffer& commands,
                                       const cv::Point2f& ego_px,
                                       bool ego_valid,
                                       const cv::Point2f& current_px,
                                       bool current_valid,
                                       const cv::Point2f& target_px,
                                       bool target_valid) {
  if (current_valid && target_valid) {
    commands.AddLine(current_px, target_px, cv::Scalar(255, 255, 255), 2.0f);
  }

  if (ego_valid) {
    commands.AddCircle(ego_px, 8.0f, cv::Scalar(0, 0, 0), 1.0f, true);
    commands.AddCircle(ego_px, 5.0f, cv::Scalar(255, 128, 255), 1.0f, true);
  }

  if (current_valid) {
    commands.AddCircle(current_px, 8.0f, cv::Scalar(0, 0, 0), 1.0f, true);
    commands.AddCircle(current_px, 6.0f, cv::Scalar(0, 255, 255), 1.0f, true);
  }

  if (target_valid) {
    commands.AddCircle(target_px, 8.0f, cv::Scalar(0, 0, 0), 1.0f, true);
    commands.AddCircle(target_px, 6.0f, cv::Scalar(255, 255, 0), 1.0f, true);
  }
}

void DrawLkaReferenceOverlayLabels(cv::Mat& image,
                                   const cv::Point2f& ego_px,
                                   bool ego_valid,
                                   const cv::Point2f& current_px,
                                   bool current_valid,
                                   const cv::Point2f& target_px,
                                   bool target_valid) {
  if (image.empty()) {
    return;
  }

  if (ego_valid) {
    DrawOutlinedText(image,
                     "Ego",
                     cv::Point(cvRound(ego_px.x + 10.0f), cvRound(ego_px.y + 18.0f)),
                     cv::Scalar(255, 128, 255));
  }

  if (current_valid) {
    DrawOutlinedText(image,
                     "LKA current",
                     cv::Point(cvRound(current_px.x + 10.0f), cvRound(current_px.y - 10.0f)),
                     cv::Scalar(0, 255, 255));
  }

  if (target_valid) {
    DrawOutlinedText(image,
                     "LKA target",
                     cv::Point(cvRound(target_px.x + 10.0f), cvRound(target_px.y - 10.0f)),
                     cv::Scalar(255, 255, 0));
  }
}

void DrawLkaReferenceOverlay(cv::Mat& image,
                             const cv::Point2f& ego_px,
                             bool ego_valid,
                             const cv::Point2f& current_px,
                             bool current_valid,
                             const cv::Point2f& target_px,
                             bool target_valid) {
  if (image.empty()) {
    return;
  }

  DrawCommandBuffer commands;
  AppendLkaReferenceOverlayCommands(commands,
                                    ego_px,
                                    ego_valid,
                                    current_px,
                                    current_valid,
                                    target_px,
                                    target_valid);
  DrawCommandsOpenCv(image, commands);
  DrawLkaReferenceOverlayLabels(image,
                                ego_px,
                                ego_valid,
                                current_px,
                                current_valid,
                                target_px,
                                target_valid);
}

void DrawPerformanceOverlay(cv::Mat& image,
                            const adas_app::RuntimePerformanceMetrics& perf) {
  if (image.empty()) {
    return;
  }

  struct PerfLine {
    const char* label;
    double value_ms;
    cv::Scalar color;
  };

  const std::vector<PerfLine> lines = {
      {"INPUT", perf.input_ms, WHITE},
      {"INFER", perf.inference_ms, CYAN},
      {"GEOM", perf.geometry_ms, ORANGE},
      {"ACC", perf.acc_ms, GREEN},
      {"LKA", perf.lka_ms, YELLOW},
      {"STAB", perf.stability_ms, WHITE},
      {"BEHAV", perf.behavior_ms, MAGENTA},
      {"COLL", perf.collision_ms, RED},
      {"DRAW", perf.overlay_ms, GRAY},
  };

  const int panel_width = 310;
  const int panel_height = 54 + static_cast<int>(lines.size()) * 22;
  const int x = std::max(8, image.cols - panel_width - 20);
  const int y = std::max(8, image.rows - panel_height - 20);

  std::ostringstream header;
  header << std::fixed << std::setprecision(1)
         << "PERF FPS:" << perf.fps
         << " TOTAL:" << std::setprecision(2) << perf.total_ms << "ms";
  DrawOutlinedText(image, header.str(), cv::Point(x + 12, y + 24), WHITE);

  int line_y = y + 48;
  for (const auto& line : lines) {
    std::ostringstream oss;
    oss << std::left << std::setw(7) << line.label
        << std::right << std::fixed << std::setprecision(2) << line.value_ms << " ms";
    DrawOutlinedText(image, oss.str(), cv::Point(x + 12, line_y), line.color);
    line_y += 22;
  }
}

}  // namespace adas_render
