#pragma once

#include <string>

#include <opencv2/core/mat.hpp>

#include "draw_commands.h"

namespace adas_render {

enum class RenderBackend {
  OpenCv,
  OpenGl
};

RenderBackend ParseRenderBackend(const std::string& backend_name);

class FramePresenter {
 public:
  FramePresenter(std::string backend_name,
                 std::string window_name,
                 int output_width,
                 int output_height,
                 int wait_key_ms);

  bool UsesOpenGl() const;
  const std::string& BackendName() const { return backend_name_; }

  int Show(cv::Mat& frame, const DrawCommandBuffer* overlay_commands = nullptr);

 private:
  int ShowOpenCv(cv::Mat& frame, const DrawCommandBuffer* overlay_commands);
  int ShowOpenGl(cv::Mat& frame, const DrawCommandBuffer* overlay_commands);

  std::string backend_name_;
  std::string window_name_;
  RenderBackend backend_;
  int output_width_;
  int output_height_;
  int wait_key_ms_;
  bool warned_fallback_ = false;
};

}  // namespace adas_render
