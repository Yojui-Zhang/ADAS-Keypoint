#pragma once

#include <opencv2/core.hpp>

#include "runtime_performance.h"
#include "draw_commands.h"

namespace adas_render {

void DrawLkaReferenceOverlay(cv::Mat& image,
                             const cv::Point2f& ego_px,
                             bool ego_valid,
                             const cv::Point2f& current_px,
                             bool current_valid,
                             const cv::Point2f& target_px,
                             bool target_valid);

void AppendLkaReferenceOverlayCommands(DrawCommandBuffer& commands,
                                       const cv::Point2f& ego_px,
                                       bool ego_valid,
                                       const cv::Point2f& current_px,
                                       bool current_valid,
                                       const cv::Point2f& target_px,
                                       bool target_valid);

void DrawLkaReferenceOverlayLabels(cv::Mat& image,
                                   const cv::Point2f& ego_px,
                                   bool ego_valid,
                                   const cv::Point2f& current_px,
                                   bool current_valid,
                                   const cv::Point2f& target_px,
                                   bool target_valid);

void DrawPerformanceOverlay(cv::Mat& image,
                            const adas_app::RuntimePerformanceMetrics& perf);

}  // namespace adas_render
