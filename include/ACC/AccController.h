#pragma once
#include "AccConfig.h"
#include "KalmanDistFiliter.h"
#include "AlphaBetaFilter.h"
#include "LeadSelector.h"
#include "GeometryAdapter.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace acc {

class AccController {
public:
  explicit AccController(AccConfig cfg = {}) : cfg_(cfg) {}

  void SetConfig(const AccConfig& cfg) { cfg_ = cfg; }
  const AccConfig& GetConfig() const { return cfg_; }

  void SetEgoSpeedKmh(float ego_speed_kmh);

  template <typename TrackingBoxT>
  AccCommand Update(const std::vector<TrackingBoxT>& world_result) {
    AccCommand out{};

    int current_frame = world_result.empty() ? -1 : world_result.front().frame;
    float dt = ComputeDtSec(current_frame);

    if (last_frame_ < 0) {
      ego_speed_est_mps_ = KmhToMps(cfg_.cruise_speed_kmh);
    }

    // 1) candidates
    std::vector<LeadCandidate> candidates;
    candidates.reserve(world_result.size());

    for (const auto& tb : world_result) {
      if (!(tb.class_id == 1 || tb.class_id == 2 || tb.class_id == 3)) continue;

      cv::Point2f ground_xy;
      if (!TryGetGroundBottomCenterXY(tb, ground_xy)) continue;

      const float forward_m = ground_xy.x;
      const float lateral_m = ground_xy.y;

      if (forward_m < cfg_.min_forward_m || forward_m > cfg_.max_forward_m) continue;
      if (std::fabs(lateral_m) > cfg_.lateral_limit_m) continue;

      candidates.push_back({tb.id, forward_m, lateral_m, tb.score});
    }

    // 2) select lead
    int lead_idx = selector_.Select(candidates, cfg_);
    const bool has_lead = (lead_idx >= 0);

    const int   lead_id        = has_lead ? candidates[lead_idx].id : -1;
    const float lead_forward_m = has_lead ? candidates[lead_idx].forward_m : 0.0f;
    const float lead_score     = has_lead ? candidates[lead_idx].score : 0.0f;

    // 3) lead filter update (✅ pass score)
    float rel_speed_mps = 0.0f;
    float filt_dist_m   = lead_forward_m;

    float dist_var = 0.0f;
    float vrel_var = 0.0f;

    if (has_lead) {
      auto& f = lead_filters_[lead_id];
      f.Update(lead_forward_m, dt, current_frame, lead_score);

      if (f.Initialized()) {
        filt_dist_m   = f.Distance();
        rel_speed_mps = f.RelSpeed();

        dist_var = f.DistanceVar();
        vrel_var = f.RelSpeedVar();
      }
    }

    // 4) longitudinal control (same as yours)
    float v_ego = std::max(0.0f, ego_speed_est_mps_);
    const float v0 = KmhToMps(cfg_.cruise_speed_kmh);
    float accel_cmd = 0.0f;

    if (!has_lead) {
      const float delta = 4.0f;
      accel_cmd = cfg_.max_accel_mps2 * (1.0f - std::pow(v_ego / std::max(0.1f, v0), delta));
    } else {
      const float closing_speed = std::max(0.0f, -rel_speed_mps);
      const float s0 = cfg_.standstill_gap_m;
      const float T  = cfg_.time_gap_s;
      const float a  = std::max(0.1f, cfg_.max_accel_mps2);
      const float b  = std::max(0.1f, cfg_.comfort_decel_mps2);

      const float s_star = s0 + v_ego * T + (v_ego * closing_speed) / (2.0f * std::sqrt(a * b));
      const float s = std::max(0.1f, filt_dist_m);

      const float delta = 4.0f;
      const float free_term   = 1.0f - std::pow(v_ego / std::max(0.1f, v0), delta);
      const float follow_term = Sqr(s_star / s);

      accel_cmd = cfg_.max_accel_mps2 * (free_term - follow_term);

      const float gap_margin = std::max(0.05f, s - s0);
      const float required_decel = (closing_speed * closing_speed) / (2.0f * gap_margin);

      if (closing_speed > 0.5f && s < (s0 + v_ego * T)) {
        accel_cmd = std::min(accel_cmd, -required_decel);
      }
    }

    // 5) clamp + jerk
    accel_cmd = Clamp(accel_cmd, -cfg_.max_decel_mps2, cfg_.max_accel_mps2);
    const float max_da = cfg_.jerk_limit_mps3 * dt;
    accel_cmd = Clamp(accel_cmd,
                      last_accel_cmd_mps2_ - max_da,
                      last_accel_cmd_mps2_ + max_da);
    last_accel_cmd_mps2_ = accel_cmd;

    // 6) update ego speed
    if (!cfg_.use_external_ego_speed) {
      v_ego = std::max(0.0f, v_ego + accel_cmd * dt);
      ego_speed_est_mps_ = v_ego;
    } else {
      v_ego = ego_speed_est_mps_;
    }

    // 7.1 outputs
    // Even when ego speed comes from CAN, output a forward-looking speed command
    // so downstream controllers can request acceleration instead of just mirroring
    // the current vehicle speed.
    const float v_cmd_mps = std::max(0.0f, v_ego + accel_cmd * dt);
    float target_speed_kmh = MpsToKmh(v_cmd_mps);
    const float brake_deadband = 0.2f;
    float brake_level = 0.0f;

    if (accel_cmd < -brake_deadband) {
      float decel_need = -accel_cmd;
      brake_level = (decel_need / std::max(0.1f, cfg_.brake_full_decel_mps2)) * cfg_.brake_multiplier;
      brake_level = Clamp(brake_level, 0.0f, 10.0f);
      target_speed_kmh = 0.0f;
    }

    out.speed_kmh = target_speed_kmh;
    out.brake_0_10 = brake_level;
    out.target_id = lead_id;
    out.target_forward_m = has_lead ? filt_dist_m : 0.0f;
    out.relative_speed_mps = has_lead ? rel_speed_mps : 0.0f;

    // 7.2 lead info + TTC (mean)
    if (has_lead) {
      const float v_lead_mps = std::max(0.0f, v_ego + rel_speed_mps);
      out.TargetSpeedKmh = MpsToKmh(v_lead_mps);
      out.Targetdistance = filt_dist_m;

      const float closing_speed = std::max(0.0f, -rel_speed_mps);
      if (closing_speed > 0.5f) out.TargetTTC = filt_dist_m / closing_speed;
      else                      out.TargetTTC = std::numeric_limits<float>::infinity();
    } else {
      out.TargetSpeedKmh = 0.0f;
      out.Targetdistance = 0.0f;
      out.TargetTTC      = std::numeric_limits<float>::infinity();
    }

    // ✅ 7.3 uncertainty export (requires AccCommand fields)
    out.TargetScore   = has_lead ? lead_score : 0.0f;
    out.TargetDistStd = has_lead ? std::sqrt(std::max(0.0f, dist_var)) : 0.0f;
    out.RelSpeedStd   = has_lead ? std::sqrt(std::max(0.0f, vrel_var)) : 0.0f;

    // TTC std propagation: ttc = d / v, where v = closing_speed = -rel_speed
    out.TargetTTCStd = 0.0f;
    if (has_lead) {
      const float v = std::max(0.0f, -rel_speed_mps);
      const float d = std::max(0.0f, filt_dist_m);

      if (v > 0.5f && d > 0.0f) {
        // var(ttc) ≈ (1/v)^2 var(d) + (d/v^2)^2 var(rel_speed)
        const float dtd = 1.0f / v;
        const float dtr = d / (v * v); // derivative wrt rel_speed (see note)
        const float ttc_var = dtd*dtd*dist_var + dtr*dtr*vrel_var;
        out.TargetTTCStd = std::sqrt(std::max(0.0f, ttc_var));
      }
    }

    last_frame_ = current_frame;
    return out;
  }

private:
  float ComputeDtSec(int current_frame);

  inline float Clamp(float v, float lo, float hi) const { return std::max(lo, std::min(v, hi)); }
  inline float KmhToMps(float kmh) const { return kmh / 3.6f; }
  inline float MpsToKmh(float mps) const { return mps * 3.6f; }
  inline float Sqr(float x) const { return x * x; }

  AccConfig cfg_;
  LeadSelector selector_;

  std::unordered_map<int, AccKalmanAdapter> lead_filters_;

  int   last_frame_ = -1;
  float ego_speed_est_mps_ = 0.0f;
  float last_accel_cmd_mps2_ = 0.0f;
};

} // namespace acc
