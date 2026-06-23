#include "draw_commands.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

namespace adas_render {

void DrawCommandBuffer::AddLine(const cv::Point2f& p0,
                                const cv::Point2f& p1,
                                const cv::Scalar& color,
                                float thickness) {
  DrawCommand cmd;
  cmd.type = DrawCommandType::Line;
  cmd.p0 = p0;
  cmd.p1 = p1;
  cmd.color = color;
  cmd.thickness = std::max(1.0f, thickness);
  commands_.push_back(cmd);
}

void DrawCommandBuffer::AddRectangle(const cv::Rect2f& rect,
                                     const cv::Scalar& color,
                                     float thickness,
                                     bool filled) {
  DrawCommand cmd;
  cmd.type = DrawCommandType::Rectangle;
  cmd.p0 = cv::Point2f(rect.x, rect.y);
  cmd.p1 = cv::Point2f(rect.x + rect.width, rect.y + rect.height);
  cmd.color = color;
  cmd.thickness = std::max(1.0f, thickness);
  cmd.filled = filled;
  commands_.push_back(cmd);
}

void DrawCommandBuffer::AddCircle(const cv::Point2f& center,
                                  float radius,
                                  const cv::Scalar& color,
                                  float thickness,
                                  bool filled) {
  if (radius <= 0.0f) {
    return;
  }

  DrawCommand cmd;
  cmd.type = DrawCommandType::Circle;
  cmd.p0 = center;
  cmd.radius = radius;
  cmd.color = color;
  cmd.thickness = std::max(1.0f, thickness);
  cmd.filled = filled;
  commands_.push_back(cmd);
}

void DrawCommandsOpenCv(cv::Mat& image, const DrawCommandBuffer& commands) {
  if (image.empty() || commands.Empty()) {
    return;
  }

  for (const DrawCommand& cmd : commands.Commands()) {
    const int thickness = cmd.filled ? cv::FILLED : static_cast<int>(std::round(cmd.thickness));
    switch (cmd.type) {
      case DrawCommandType::Line:
        cv::line(image, cmd.p0, cmd.p1, cmd.color, thickness, cv::LINE_AA);
        break;
      case DrawCommandType::Rectangle: {
        const cv::Rect rect(cvRound(cmd.p0.x),
                            cvRound(cmd.p0.y),
                            cvRound(cmd.p1.x - cmd.p0.x),
                            cvRound(cmd.p1.y - cmd.p0.y));
        cv::rectangle(image, rect, cmd.color, thickness, cv::LINE_AA);
        break;
      }
      case DrawCommandType::Circle:
        cv::circle(image, cmd.p0, cvRound(cmd.radius), cmd.color, thickness, cv::LINE_AA);
        break;
    }
  }
}

}  // namespace adas_render
