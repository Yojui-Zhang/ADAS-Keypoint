#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace adas_render {

enum class DrawCommandType {
  Line,
  Rectangle,
  Circle
};

struct DrawCommand {
  DrawCommandType type = DrawCommandType::Line;
  cv::Point2f p0;
  cv::Point2f p1;
  float radius = 0.0f;
  cv::Scalar color = cv::Scalar(255, 255, 255);
  float thickness = 1.0f;
  bool filled = false;
};

class DrawCommandBuffer {
 public:
  void Clear() { commands_.clear(); }
  bool Empty() const { return commands_.empty(); }
  const std::vector<DrawCommand>& Commands() const { return commands_; }

  void AddLine(const cv::Point2f& p0,
               const cv::Point2f& p1,
               const cv::Scalar& color,
               float thickness = 1.0f);

  void AddRectangle(const cv::Rect2f& rect,
                    const cv::Scalar& color,
                    float thickness = 1.0f,
                    bool filled = false);

  void AddCircle(const cv::Point2f& center,
                 float radius,
                 const cv::Scalar& color,
                 float thickness = 1.0f,
                 bool filled = false);

 private:
  std::vector<DrawCommand> commands_;
};

void DrawCommandsOpenCv(cv::Mat& image, const DrawCommandBuffer& commands);

}  // namespace adas_render
