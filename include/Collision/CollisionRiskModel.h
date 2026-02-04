#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>

namespace collision {
namespace risk {

// 評估結果：用來避免「非緊急也觸發」與「車已經行駛過去仍觸發」。
// 核心：必須同時滿足 (1) 會在預測期內進入 ego path corridor, (2) 有明顯的接近速度(approach speed)。
struct ThreatEval {
  bool valid = false;

  float t_hit_s = std::numeric_limits<float>::infinity();      // 第一次進入 corridor 的時間
  float min_dist_m = std::numeric_limits<float>::infinity();   // 在 0~danger_forward 內對 ego_path 的最小距離
  float dist_now_m = std::numeric_limits<float>::infinity();   // 當下距離(對 ego_path)
  float approach_speed_mps = 0.f;                               // -(p·v)/|p|, >0 代表接近

  float score = std::numeric_limits<float>::infinity();        // 用於排序：越小越危險
  cv::Point2f hit_pos{};
  std::string reason;
};

static inline float dot2(const cv::Point2f& a, const cv::Point2f& b) {
  return a.x * b.x + a.y * b.y;
}
static inline float norm2(const cv::Point2f& a) {
  return std::sqrt(std::max(0.f, dot2(a, a)));
}

static inline float MinDistToPath(const cv::Point2f& p,
                                  const std::vector<cv::Point2f>& path)
{
  float best = std::numeric_limits<float>::infinity();
  for (const auto& q : path) {
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    const float d = std::sqrt(dx*dx + dy*dy);
    if (d < best) best = d;
  }
  return best;
}

// 常速(CV)預測 + corridor 侵入時間(類 TTC)。
// - attention_half_width_m：前置關注帶(比 corridor 寬)，超出直接忽略，降低隔壁車道誤觸發
// - min_approach_speed_mps：最小接近速度門檻；低於此值視為「在遠離或擦身而過」
static inline ThreatEval EvaluateConstantVelocityCorridorRisk(
    const cv::Point2f& p0,
    const cv::Point2f& vrel,
    const std::vector<cv::Point2f>& pred,
    const std::vector<cv::Point2f>& ego_path,
    float danger_forward_m,
    float corridor_half_width_m,
    float attention_half_width_m,
    float min_approach_speed_mps,
    float step_s)
{
  ThreatEval ev;

  if (ego_path.empty() || pred.empty()) {
    ev.reason = "empty path/pred";
    return ev;
  }

  // 0) 當下距離：先做「關注帶」過濾
  ev.dist_now_m = MinDistToPath(p0, ego_path);
  if (!std::isfinite(ev.dist_now_m) || ev.dist_now_m > attention_half_width_m) {
    ev.reason = "outside attention band";
    return ev;
  }

  // 1) 接近速度：避免「車已經過去仍觸發」
  const float pnorm = std::max(1e-3f, norm2(p0));
  ev.approach_speed_mps = -dot2(p0, vrel) / pnorm; // >0 => approaching
  if (!std::isfinite(ev.approach_speed_mps) || ev.approach_speed_mps < min_approach_speed_mps) {
    ev.reason = "low approach speed";
    return ev;
  }

  // 2) 掃描預測點：找 first_hit_t + min_dist
  const float dt = std::max(step_s, 1e-3f);
  float min_dist = std::numeric_limits<float>::infinity();
  float first_hit_t = std::numeric_limits<float>::infinity();
  cv::Point2f hit_pos{};

  for (size_t i = 0; i < pred.size(); ++i) {
    const float t = static_cast<float>(i) * dt;
    const auto& p = pred[i];

    if (p.x < 0.f || p.x > danger_forward_m) continue;

    const float d = MinDistToPath(p, ego_path);
    if (d < min_dist) min_dist = d;

    if (d <= corridor_half_width_m) {
      first_hit_t = t;
      hit_pos = p;
      break;
    }
  }

  if (!std::isfinite(min_dist)) {
    ev.reason = "no valid point in forward window";
    return ev;
  }
  if (!std::isfinite(first_hit_t)) {
    ev.reason = "no corridor hit";
    return ev;
  }

  ev.valid = true;
  ev.t_hit_s = first_hit_t;
  ev.min_dist_m = min_dist;
  ev.hit_pos = hit_pos;

  // score：以 t_hit 為主，min_dist 為輔
  ev.score = ev.t_hit_s + 0.15f * ev.min_dist_m;
  ev.reason = "ok";
  return ev;
}

} // namespace risk
} // namespace collision

