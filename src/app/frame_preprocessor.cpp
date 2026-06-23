#include "frame_preprocessor.h"

#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>

#include "config.h"

namespace adas_app {

bool PrepareProcessFrame(const cv::Mat& input_view, cv::Mat& process_frame) {
  if (input_view.empty()) return false;

  const int roi_w = std::min(rect_video_width, input_view.cols);
  const int roi_h = std::min(rect_video_height, input_view.rows);
  const int roi_x = std::max(0, input_view.cols - roi_w);
  const int roi_y = std::max(0, input_view.rows - roi_h);

  const cv::Rect roi_input_view(roi_x, roi_y, roi_w, roi_h);
  cv::Mat roi = input_view(roi_input_view);
  cv::resize(roi, process_frame, cv::Size(process_video_width, process_video_height));
  return true;
}

}  // namespace adas_app
