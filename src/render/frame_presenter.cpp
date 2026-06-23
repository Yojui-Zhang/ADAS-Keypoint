#include "frame_presenter.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _opengl
#include <GLES2/gl2.h>

extern void imageShow(int width, int height, unsigned char rgb[]);
extern void swap_egl(void);
#endif

namespace {

std::string ToLowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

#ifdef _opengl
struct PrimitiveGlState {
  GLuint program = 0;
  GLint position_attr = -1;
  GLint color_uniform = -1;
  bool initialized = false;
};

GLuint CompileShader(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled != GL_TRUE) {
    char log[512] = {};
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    std::cerr << "FramePresenter: primitive shader compile failed: " << log << std::endl;
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

bool InitPrimitiveGlState(PrimitiveGlState& state) {
  if (state.initialized) {
    return state.program != 0;
  }

  static const char* kVertexShader =
      "attribute vec2 a_position;\n"
      "void main() {\n"
      "  gl_Position = vec4(a_position, 0.0, 1.0);\n"
      "}\n";
  static const char* kFragmentShader =
      "precision mediump float;\n"
      "uniform vec4 u_color;\n"
      "void main() {\n"
      "  gl_FragColor = u_color;\n"
      "}\n";

  state.initialized = true;
  const GLuint vertex_shader = CompileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fragment_shader = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
  if (vertex_shader == 0 || fragment_shader == 0) {
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return false;
  }

  state.program = glCreateProgram();
  glAttachShader(state.program, vertex_shader);
  glAttachShader(state.program, fragment_shader);
  glLinkProgram(state.program);
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  GLint linked = GL_FALSE;
  glGetProgramiv(state.program, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    char log[512] = {};
    glGetProgramInfoLog(state.program, sizeof(log), nullptr, log);
    std::cerr << "FramePresenter: primitive program link failed: " << log << std::endl;
    glDeleteProgram(state.program);
    state.program = 0;
    return false;
  }

  state.position_attr = glGetAttribLocation(state.program, "a_position");
  state.color_uniform = glGetUniformLocation(state.program, "u_color");
  return state.position_attr >= 0 && state.color_uniform >= 0;
}

void AppendPixelPoint(std::vector<float>& vertices, float x, float y, float width, float height) {
  const float ndc_x = (x / std::max(1.0f, width)) * 2.0f - 1.0f;
  const float ndc_y = 1.0f - (y / std::max(1.0f, height)) * 2.0f;
  vertices.push_back(ndc_x);
  vertices.push_back(ndc_y);
}

void SetPrimitiveColor(GLint color_uniform, const cv::Scalar& color) {
  glUniform4f(color_uniform,
              static_cast<float>(color[2]) / 255.0f,
              static_cast<float>(color[1]) / 255.0f,
              static_cast<float>(color[0]) / 255.0f,
              static_cast<float>(color[3] == 0.0 ? 255.0 : color[3]) / 255.0f);
}

void DrawPrimitiveVertices(GLenum mode,
                           const std::vector<float>& vertices,
                           float thickness,
                           const cv::Scalar& color,
                           PrimitiveGlState& state,
                           float line_scale) {
  if (vertices.empty()) {
    return;
  }

  SetPrimitiveColor(state.color_uniform, color);
  glLineWidth(std::max(1.0f, thickness * line_scale));
  glVertexAttribPointer(state.position_attr, 2, GL_FLOAT, GL_FALSE, 0, vertices.data());
  glEnableVertexAttribArray(state.position_attr);
  glDrawArrays(mode, 0, static_cast<GLsizei>(vertices.size() / 2));
}

void DrawOpenGlCommands(const adas_render::DrawCommandBuffer& commands,
                        int source_width,
                        int source_height,
                        int output_width,
                        int output_height) {
  if (commands.Empty()) {
    return;
  }

  static PrimitiveGlState state;
  if (!InitPrimitiveGlState(state)) {
    return;
  }

  const float source_w = static_cast<float>(std::max(1, source_width));
  const float source_h = static_cast<float>(std::max(1, source_height));
  const float line_scale =
      0.5f * (static_cast<float>(output_width) / source_w + static_cast<float>(output_height) / source_h);

  glUseProgram(state.program);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  for (const adas_render::DrawCommand& cmd : commands.Commands()) {
    std::vector<float> vertices;
    switch (cmd.type) {
      case adas_render::DrawCommandType::Line:
        AppendPixelPoint(vertices, cmd.p0.x, cmd.p0.y, source_w, source_h);
        AppendPixelPoint(vertices, cmd.p1.x, cmd.p1.y, source_w, source_h);
        DrawPrimitiveVertices(GL_LINES, vertices, cmd.thickness, cmd.color, state, line_scale);
        break;

      case adas_render::DrawCommandType::Rectangle: {
        const float x0 = cmd.p0.x;
        const float y0 = cmd.p0.y;
        const float x1 = cmd.p1.x;
        const float y1 = cmd.p1.y;
        if (cmd.filled) {
          AppendPixelPoint(vertices, x0, y0, source_w, source_h);
          AppendPixelPoint(vertices, x1, y0, source_w, source_h);
          AppendPixelPoint(vertices, x0, y1, source_w, source_h);
          AppendPixelPoint(vertices, x1, y1, source_w, source_h);
          DrawPrimitiveVertices(GL_TRIANGLE_STRIP, vertices, cmd.thickness, cmd.color, state, line_scale);
        } else {
          AppendPixelPoint(vertices, x0, y0, source_w, source_h);
          AppendPixelPoint(vertices, x1, y0, source_w, source_h);
          AppendPixelPoint(vertices, x1, y1, source_w, source_h);
          AppendPixelPoint(vertices, x0, y1, source_w, source_h);
          DrawPrimitiveVertices(GL_LINE_LOOP, vertices, cmd.thickness, cmd.color, state, line_scale);
        }
        break;
      }

      case adas_render::DrawCommandType::Circle: {
        constexpr double kPi = 3.14159265358979323846;
        const int segments = std::max(24, std::min(96, static_cast<int>(cmd.radius * 2.0f)));
        if (cmd.filled) {
          AppendPixelPoint(vertices, cmd.p0.x, cmd.p0.y, source_w, source_h);
        }
        for (int i = 0; i < segments; ++i) {
          const float theta = static_cast<float>(2.0 * kPi * static_cast<double>(i) /
                                                 static_cast<double>(segments));
          const float x = cmd.p0.x + std::cos(theta) * cmd.radius;
          const float y = cmd.p0.y + std::sin(theta) * cmd.radius;
          AppendPixelPoint(vertices, x, y, source_w, source_h);
        }
        if (cmd.filled) {
          const float x = cmd.p0.x + cmd.radius;
          AppendPixelPoint(vertices, x, cmd.p0.y, source_w, source_h);
          DrawPrimitiveVertices(GL_TRIANGLE_FAN, vertices, cmd.thickness, cmd.color, state, line_scale);
        } else {
          DrawPrimitiveVertices(GL_LINE_LOOP, vertices, cmd.thickness, cmd.color, state, line_scale);
        }
        break;
      }
    }
  }

  glDisableVertexAttribArray(state.position_attr);
  glLineWidth(1.0f);
}
#endif

}  // namespace

namespace adas_render {

RenderBackend ParseRenderBackend(const std::string& backend_name) {
  const std::string backend = ToLowerCopy(backend_name);
  if (backend == "opengl" || backend == "gl" || backend == "gpu") {
    return RenderBackend::OpenGl;
  }
  return RenderBackend::OpenCv;
}

FramePresenter::FramePresenter(std::string backend_name,
                               std::string window_name,
                               int output_width,
                               int output_height,
                               int wait_key_ms)
    : backend_name_(std::move(backend_name)),
      window_name_(std::move(window_name)),
      backend_(ParseRenderBackend(backend_name_)),
      output_width_(output_width),
      output_height_(output_height),
      wait_key_ms_(wait_key_ms) {}

bool FramePresenter::UsesOpenGl() const {
#ifdef _opengl
  return backend_ == RenderBackend::OpenGl;
#else
  return false;
#endif
}

int FramePresenter::Show(cv::Mat& frame, const DrawCommandBuffer* overlay_commands) {
  if (backend_ == RenderBackend::OpenGl) {
    return ShowOpenGl(frame, overlay_commands);
  }
  return ShowOpenCv(frame, overlay_commands);
}

int FramePresenter::ShowOpenCv(cv::Mat& frame, const DrawCommandBuffer* overlay_commands) {
  if (overlay_commands != nullptr) {
    DrawCommandsOpenCv(frame, *overlay_commands);
  }
  cv::resize(frame, frame, cv::Size(output_width_, output_height_));
  cv::imshow(window_name_, frame);
  return cv::waitKey(wait_key_ms_);
}

int FramePresenter::ShowOpenGl(cv::Mat& frame, const DrawCommandBuffer* overlay_commands) {
#ifdef _opengl
  const int source_width = frame.cols;
  const int source_height = frame.rows;
  cv::resize(frame, frame, cv::Size(output_width_, output_height_));
  imageShow(output_width_, output_height_, frame.data);
  if (overlay_commands != nullptr) {
    DrawOpenGlCommands(*overlay_commands, source_width, source_height, output_width_, output_height_);
  }
  swap_egl();
  return cv::waitKey(wait_key_ms_);
#else
  if (!warned_fallback_) {
    std::cerr << "FramePresenter: render_backend=opengl requested, "
              << "but this binary was built without _opengl. Fallback to OpenCV."
              << std::endl;
    warned_fallback_ = true;
  }
  return ShowOpenCv(frame, overlay_commands);
#endif
}

}  // namespace adas_render
