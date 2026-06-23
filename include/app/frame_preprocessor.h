#pragma once

#include <opencv2/core/mat.hpp>

namespace adas_app {

bool PrepareProcessFrame(const cv::Mat& input_view, cv::Mat& process_frame);

}  // namespace adas_app
