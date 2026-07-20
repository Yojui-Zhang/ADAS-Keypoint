#pragma once

#include "AccConfig.h"
#include "GeometryAdapter.h"
#include "config.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace acc {

inline cv::Scalar AccObjectStateColor(AccTrackedObjectState state) {
  switch (state) {
    case AccTrackedObjectState::FollowingLead: return GREEN;
    case AccTrackedObjectState::Lead: return CYAN;
    case AccTrackedObjectState::Candidate: return ORANGE;
    default: return GRAY;
  }
}

inline const char* AccObjectStateOverlayLabel(AccTrackedObjectState state) {
  switch (state) {
    case AccTrackedObjectState::FollowingLead: return "FOLLOW";
    case AccTrackedObjectState::Lead: return "LEAD";
    case AccTrackedObjectState::Candidate: return "CAND";
    default: return "REM";
  }
}

inline cv::Scalar AccLongitudinalPhaseColor(AccLongitudinalPhase phase) {
  switch (phase) {
    case AccLongitudinalPhase::MaxHold: return GREEN;
    case AccLongitudinalPhase::Accelerating: return CYAN;
    case AccLongitudinalPhase::Coasting: return ORANGE;
    case AccLongitudinalPhase::Idle: return YELLOW;
    case AccLongitudinalPhase::Braking: return RED;
  }
  return WHITE;
}

inline const char* AccLongitudinalPhaseOverlayLabel(AccLongitudinalPhase phase) {
  switch (phase) {
    case AccLongitudinalPhase::MaxHold: return "MAX HOLD";
    case AccLongitudinalPhase::Accelerating: return "ACCEL";
    case AccLongitudinalPhase::Coasting: return "COAST";
    case AccLongitudinalPhase::Idle: return "IDLE";
    case AccLongitudinalPhase::Braking: return "BRAKE";
  }
  return "UNKNOWN";
}

inline void DrawAccOutlinedText(cv::Mat& frame,
                                const std::string& text,
                                const cv::Point& origin,
                                const cv::Scalar& color,
                                double scale = 0.5,
                                int thickness = 1) {
  cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 2, cv::LINE_AA);
  cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

inline void ACC_DrawTrackingBoxes(cv::Mat& frame,
                                  const std::vector<TrackingBox>& world_result,
                                  const AccCommand& cmd,
                                  int thickness = 2) {
  if (frame.empty()) return;

  for (const auto& tb : world_result) {
    if (!(tb.class_id == 1 || tb.class_id == 2 || tb.class_id == 3)) continue;

    const AccTrackedObjectState state = ClassifyAccTrackedObjectState(cmd, tb.id);
    const bool target = (tb.id == cmd.target_id);

    const std::string state_text = AccObjectStateOverlayLabel(state);
    const cv::Scalar color = AccObjectStateColor(state);

    int box_thickness = std::max(1, thickness - 1);
    if (state == AccTrackedObjectState::Lead || state == AccTrackedObjectState::FollowingLead) {
      box_thickness = thickness + 1;
    } else if (state == AccTrackedObjectState::Candidate) {
      box_thickness = thickness;
    }

    cv::rectangle(frame, tb.box, color, box_thickness);

    cv::Point2f ground_xy;
    float dist_m = 0.0f;
    float lateral_m = 0.0f;
    if (TryGetGroundBottomCenterXY(tb, ground_xy)) {
      dist_m = ground_xy.x;
      lateral_m = ground_xy.y;
    }

    std::ostringstream oss;
    oss << state_text
        << " id:" << tb.id << " cls:" << tb.class_id
        << " x:" << std::fixed << std::setprecision(1) << dist_m << "m"
        << " y:" << std::fixed << std::setprecision(1) << lateral_m << "m";

    if (target) {
      if (std::isfinite(cmd.TargetTTC)) oss << " TTC:" << std::fixed << std::setprecision(1) << cmd.TargetTTC << "s";
      else                             oss << " TTC:inf";
      oss << " v_lead:" << std::fixed << std::setprecision(1) << cmd.TargetSpeedKmh << "km/h";
    }

    const cv::Point org(tb.box.x, std::max(0, tb.box.y - 6));
    DrawAccOutlinedText(frame, oss.str(), org, color);
  }
}

inline void ACC_DrawLongitudinalPhaseHud(cv::Mat& frame,
                                         const AccCommand& cmd) {
  if (frame.empty()) return;

  struct PhaseRow {
    AccLongitudinalPhase phase;
    const char* label;
  };

  const std::array<PhaseRow, 5> rows = {{{AccLongitudinalPhase::MaxHold, "1. MAX HOLD"},
                                         {AccLongitudinalPhase::Accelerating, "2. ACCEL"},
                                         {AccLongitudinalPhase::Coasting, "3. COAST"},
                                         {AccLongitudinalPhase::Idle, "4. IDLE"},
                                         {AccLongitudinalPhase::Braking, "5. BRAKE"}}};

  const int x = 8;
  const int y = 120;

  // cv::rectangle(frame, panel, cv::Scalar(36, 36, 36), cv::FILLED, cv::LINE_AA);
  // cv::rectangle(frame, panel, WHITE, 1, cv::LINE_AA);
  DrawAccOutlinedText(frame, "ACC SPEED PHASE", cv::Point(x + 12, y + 24), WHITE, 0.55, 1);

  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    const bool active = (cmd.longitudinal_phase == rows[i].phase);
    const cv::Scalar phase_color = AccLongitudinalPhaseColor(rows[i].phase);
    const cv::Scalar bullet_color = active ? phase_color : cv::Scalar(90, 90, 90);
    const cv::Scalar text_color = active ? phase_color : cv::Scalar(170, 170, 170);
    const int line_y = y + 48 + i * 20;
    const cv::Rect bullet_rect(x + 12, line_y - 11, 12, 12);

    cv::rectangle(frame, bullet_rect, bullet_color, cv::FILLED, cv::LINE_AA);
    if (active) {
      cv::rectangle(frame, bullet_rect, WHITE, 1, cv::LINE_AA);
    }

    std::string line = active ? std::string("> ") + rows[i].label : std::string("  ") + rows[i].label;
    DrawAccOutlinedText(frame, line, cv::Point(x + 32, line_y), text_color, 0.52, 1);
  }
}

} // namespace acc
