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
  explicit AccController(AccConfig cfg = {}) : cfg_(cfg), stop_and_go_(cfg.stop_and_go) {}

  void SetConfig(const AccConfig& cfg) {
    cfg_ = cfg;
    stop_and_go_.SetConfig(cfg_.stop_and_go);
  }
  const AccConfig& GetConfig() const { return cfg_; }

  void SetEgoSpeedKmh(float ego_speed_kmh);
  void RequestManualResume() noexcept { stop_and_go_.RequestManualResume(); }

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

    out.candidate_ids.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      out.candidate_ids.push_back(candidate.id);
    }

    // 2) select lead
    int lead_idx = selector_.Select(candidates, cfg_);
    const bool has_lead = (lead_idx >= 0);

    const int   lead_id        = has_lead ? candidates[lead_idx].id : -1;
    const float lead_forward_m = has_lead ? candidates[lead_idx].forward_m : 0.0f;
    const float lead_lateral_m = has_lead ? candidates[lead_idx].lateral_m : 0.0f;
    const float lead_score     = has_lead ? candidates[lead_idx].score : 0.0f;

    out.has_lead = has_lead;
    out.cruise_speed_kmh = cfg_.cruise_speed_kmh;
    out.lead_lateral_m = lead_lateral_m;

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
    const float ego_speed_input_mps = v_ego;
    out.ego_speed_kmh = MpsToKmh(ego_speed_input_mps);
    const float v0 = KmhToMps(cfg_.cruise_speed_kmh);
    constexpr float kAccelInfluenceEpsMps2 = 0.05f;
    const float delta = 4.0f;
    const float free_accel_nom =
        cfg_.max_accel_mps2 * (1.0f - std::pow(v_ego / std::max(0.1f, v0), delta));
    float accel_cmd = free_accel_nom;
    bool lead_coast_guard = false;
    bool lead_brake_guard = false;
    bool lead_hard_brake_guard = false;
    float closing_speed_mps = 0.0f;
    float desired_gap_m = 0.0f;
    float coast_gap_m = 0.0f;
    float high_speed_brake_gap_m = 0.0f;
    float follow_ttc_s = std::numeric_limits<float>::infinity();
    float gap_ratio = 0.0f;
    bool safe_speed_hold = false;

    if (has_lead) {
      closing_speed_mps = std::max(0.0f, -rel_speed_mps);
      const float s0 = cfg_.standstill_gap_m;
      const float T  = cfg_.time_gap_s;
      const float a  = std::max(0.1f, cfg_.max_accel_mps2);
      const float b  = std::max(0.1f, cfg_.comfort_decel_mps2);

      const float s_star = s0 + v_ego * T + (v_ego * closing_speed_mps) / (2.0f * std::sqrt(a * b));
      const float s = std::max(0.1f, filt_dist_m);
      desired_gap_m = s0 + v_ego * T;
      gap_ratio = s / std::max(0.1f, desired_gap_m);
      const bool high_speed_relax =
          cfg_.high_speed_relax_enable &&
          MpsToKmh(v_ego) >= std::max(0.0f, cfg_.high_speed_relax_min_kmh);
      const bool closing_fast =
          closing_speed_mps >= std::max(0.0f, cfg_.high_speed_brake_closing_mps);
      high_speed_brake_gap_m =
          s0 +
          v_ego * std::max(0.0f, cfg_.high_speed_brake_time_gap_s) +
          std::max(0.0f, cfg_.high_speed_brake_gap_margin_m);

      const float free_term   = 1.0f - std::pow(v_ego / std::max(0.1f, v0), delta);
      const float follow_term = Sqr(s_star / s);

      accel_cmd = cfg_.max_accel_mps2 * (free_term - follow_term);

      const float gap_margin = std::max(0.05f, s - s0);
      const float required_decel = (closing_speed_mps * closing_speed_mps) / (2.0f * gap_margin);

      if (closing_speed_mps > 0.5f && s < (s0 + v_ego * T)) {
        accel_cmd = std::min(accel_cmd, -required_decel);
      }

      coast_gap_m =
          desired_gap_m +
          std::max(0.0f, cfg_.coast_gap_margin_m) +
          v_ego * std::max(0.0f, cfg_.coast_time_gap_margin_s);
      const bool suppress_gap_brake =
          high_speed_relax && !closing_fast && s > high_speed_brake_gap_m;

      if (suppress_gap_brake) {
        if (s <= coast_gap_m) {
          lead_coast_guard = true;
          accel_cmd = 0.0f;
        } else {
          accel_cmd = free_accel_nom;
        }
      } else if (s <= coast_gap_m) {
        lead_coast_guard = true;
        accel_cmd = std::min(accel_cmd, 0.0f);
      }

      const float brake_gap = desired_gap_m + std::max(0.0f, cfg_.brake_gap_margin_m);
      if (!suppress_gap_brake && s <= brake_gap) {
        const float gap_deficit_m = std::max(0.0f, brake_gap - s);
        float gap_decel = gap_deficit_m *
            std::max(0.0f, cfg_.gap_error_decel_gain_mps2_per_m);
        gap_decel = Clamp(gap_decel,
                          std::max(0.0f, cfg_.min_brake_decel_mps2),
                          std::max(0.1f, cfg_.comfort_decel_mps2));
        lead_brake_guard = true;
        accel_cmd = std::min(accel_cmd, -gap_decel);
      }

      if (!suppress_gap_brake && closing_speed_mps > 0.5f) {
        const float ttc_s = s / closing_speed_mps;
        follow_ttc_s = ttc_s;
        const float soft_ttc = std::max(0.1f, cfg_.ttc_soft_brake_s);
        const float hard_ttc = std::max(0.1f, cfg_.ttc_hard_brake_s);
        if (ttc_s <= soft_ttc) {
          const float span = std::max(0.1f, soft_ttc - hard_ttc);
          const float alpha = Clamp((soft_ttc - ttc_s) / span, 0.0f, 1.0f);
          const float min_decel = std::max(0.0f, cfg_.min_brake_decel_mps2);
          const float ttc_decel =
              min_decel + alpha * (std::max(min_decel, cfg_.comfort_decel_mps2) - min_decel);
          lead_brake_guard = true;
          accel_cmd = std::min(accel_cmd, -ttc_decel);
        }
        if (ttc_s <= hard_ttc) {
          lead_hard_brake_guard = true;
          accel_cmd = std::min(accel_cmd, -std::max(0.1f, cfg_.max_decel_mps2));
        }
      }

      safe_speed_hold =
          cfg_.speed_hold_enable &&
          MpsToKmh(v_ego) >= std::max(0.0f, cfg_.speed_hold_min_speed_kmh) &&
          closing_speed_mps <= std::max(0.0f, cfg_.speed_hold_max_closing_mps) &&
          (!std::isfinite(follow_ttc_s) ||
           follow_ttc_s >= std::max(0.0f, cfg_.speed_hold_min_ttc_s)) &&
          gap_ratio >= std::max(0.0f, cfg_.speed_hold_min_gap_ratio) &&
          !lead_brake_guard &&
          !lead_hard_brake_guard;
    }

    const bool lead_changed =
        has_lead &&
        lead_id != previous_lead_id_;

    if (lead_changed) {
      cut_in_elapsed_s_ = 0.0f;
      cut_in_start_accel_mps2_ = last_accel_cmd_mps2_;
    }

    previous_lead_id_ = has_lead ? lead_id : -1;

    const bool emergency_cut_in =
        has_lead &&
        ((std::isfinite(follow_ttc_s) &&
          follow_ttc_s <= std::max(0.0f, cfg_.cut_in_emergency_ttc_s)) ||
         closing_speed_mps >= std::max(0.0f, cfg_.cut_in_emergency_closing_mps) ||
         gap_ratio <= std::max(0.0f, cfg_.cut_in_emergency_gap_ratio));

    if (cfg_.cut_in_blend_enable &&
        has_lead &&
        !emergency_cut_in &&
        cut_in_elapsed_s_ < std::max(0.0f, cfg_.cut_in_blend_time_s)) {
      cut_in_elapsed_s_ += dt;
      const float blend =
          Clamp(cut_in_elapsed_s_ /
                    std::max(0.05f, cfg_.cut_in_blend_time_s),
                0.0f,
                1.0f);
      accel_cmd =
          cut_in_start_accel_mps2_ +
          blend * (accel_cmd - cut_in_start_accel_mps2_);
      out.cut_in_transition_active = true;
      out.cut_in_blend = blend;
    }

    out.desired_gap_m = desired_gap_m;
    out.coast_gap_m = coast_gap_m;
    out.high_speed_brake_gap_m = high_speed_brake_gap_m;
    out.closing_speed_mps = closing_speed_mps;
    out.gap_ratio = gap_ratio;

    const float lead_speed_mps =
        has_lead ? std::max(0.0f, v_ego + rel_speed_mps) : 0.0f;

    StopAndGoInput stop_input{};
    stop_input.has_lead = has_lead;
    stop_input.lead_id = lead_id;
    stop_input.ego_speed_mps = v_ego;
    stop_input.lead_speed_mps = lead_speed_mps;
    stop_input.distance_m = has_lead ? filt_dist_m : 0.0f;
    stop_input.standstill_gap_m = cfg_.standstill_gap_m;
    stop_input.cruise_speed_kmh = cfg_.cruise_speed_kmh;
    stop_input.base_accel_mps2 = accel_cmd;
    stop_input.dt_s = dt;

    const StopAndGoOutput stop_output = stop_and_go_.Update(stop_input);
    accel_cmd = stop_output.accel_cmd_mps2;

    out.stop_state = stop_output.state;
    out.stop_state_time_s = stop_output.state_time_s;
    out.stop_hold_active = stop_output.force_hold_brake;
    out.resume_active = stop_output.state == AccStopState::Resuming;
    out.resume_without_lead_active = stop_output.resume_without_lead_active;
    out.held_lead_id = stop_output.held_lead_id;
    out.held_lead_distance_m = stop_output.held_lead_distance_m;
    out.resume_confirm_time_s = stop_output.resume_confirm_time_s;
    out.stop_output_accel_mps2 = stop_output.accel_cmd_mps2;
    out.hold_brake_0_10 = stop_output.hold_brake_0_10;

    // 5) clamp + jerk
    const bool force_stationary_hold = stop_output.force_hold_brake;
    const float max_da = std::max(0.0f, cfg_.jerk_limit_mps3) * dt;
    const float free_accel_limited = Clamp(free_accel_nom,
                                           last_accel_cmd_mps2_ - max_da,
                                           last_accel_cmd_mps2_ + max_da);
    if (!force_stationary_hold) {
      accel_cmd = Clamp(accel_cmd, -cfg_.max_decel_mps2, cfg_.max_accel_mps2);
      accel_cmd = Clamp(accel_cmd,
                        last_accel_cmd_mps2_ - max_da,
                        last_accel_cmd_mps2_ + max_da);
      if (lead_coast_guard) {
        accel_cmd = std::min(accel_cmd, 0.0f);
      }
      if (lead_brake_guard) {
        accel_cmd = std::min(accel_cmd, -std::max(0.0f, cfg_.min_brake_decel_mps2));
      }
      if (lead_hard_brake_guard) {
        accel_cmd = std::min(accel_cmd, -std::max(0.1f, cfg_.max_decel_mps2));
      }
      accel_cmd = Clamp(accel_cmd, -cfg_.max_decel_mps2, cfg_.max_accel_mps2);
    } else {
      accel_cmd = 0.0f;
    }

    const bool stop_and_go_active = out.stop_state != AccStopState::Moving;
    out.lead_following_active =
        stop_and_go_active ||
        (has_lead && ((free_accel_limited - accel_cmd) > kAccelInfluenceEpsMps2 || accel_cmd < -0.05f));
    out.accel_cmd_mps2 = accel_cmd;
    out.free_accel_nom_mps2 = free_accel_nom;
    out.free_accel_limited_mps2 = free_accel_limited;

    last_accel_cmd_mps2_ = force_stationary_hold ? 0.0f : accel_cmd;

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
    const float throttle_deadband = std::max(0.0f, cfg_.throttle_accel_deadband_mps2);
    const float brake_deadband = std::max(0.0f, cfg_.brake_accel_deadband_mps2);
    const float estimated_lead_speed_kmh =
        has_lead ? MpsToKmh(std::max(0.0f, v_ego + rel_speed_mps)) : 0.0f;
    float target_speed_kmh = 0.0f;
    float brake_level = 0.0f;

    if (accel_cmd < -brake_deadband) {
      const float decel_need_mps2 = -accel_cmd;
      const float full_decel_mps2 = std::max(0.1f, cfg_.brake_full_decel_mps2);
      brake_level = (decel_need_mps2 / full_decel_mps2) * cfg_.brake_multiplier;
      brake_level = Clamp(brake_level, 0.0f, 10.0f);
    }

    const bool has_brake_command = brake_level > 1e-3f;
    const bool coasting_phase =
        has_lead &&
        out.lead_following_active &&
        !has_brake_command &&
        accel_cmd <= throttle_deadband;
    out.speed_hold_recommended = coasting_phase && safe_speed_hold;

    if (has_brake_command) {
      target_speed_kmh = 0.0f;
    } else if (coasting_phase) {
      target_speed_kmh =
          std::min(std::max(0.0f, out.ego_speed_kmh),
                   std::max(0.0f, cfg_.cruise_speed_kmh));
    } else if (!has_lead || !out.lead_following_active) {
      target_speed_kmh = std::max(0.0f, cfg_.cruise_speed_kmh);
    } else {
      target_speed_kmh = MpsToKmh(v_cmd_mps);
    }

    const bool braking_phase = has_brake_command;
    const bool accelerating_phase =
        !braking_phase &&
        !coasting_phase &&
        target_speed_kmh > 0.2f &&
        target_speed_kmh > out.ego_speed_kmh + 0.2f;
    const bool max_hold_phase =
        !braking_phase &&
        !coasting_phase &&
        !accelerating_phase &&
        target_speed_kmh > 0.2f;

    if (braking_phase) out.longitudinal_phase = AccLongitudinalPhase::Braking;
    else if (coasting_phase) out.longitudinal_phase = AccLongitudinalPhase::Coasting;
    else if (accelerating_phase) out.longitudinal_phase = AccLongitudinalPhase::Accelerating;
    else if (max_hold_phase) out.longitudinal_phase = AccLongitudinalPhase::MaxHold;
    else out.longitudinal_phase = AccLongitudinalPhase::Idle;

    if (stop_output.force_hold_brake) {
      out.speed_hold_recommended = false;
      target_speed_kmh = 0.0f;
      brake_level = std::max(brake_level, stop_output.hold_brake_0_10);
      out.longitudinal_phase = AccLongitudinalPhase::Braking;
    } else if (stop_output.state == AccStopState::Resuming) {
      out.speed_hold_recommended = false;
      target_speed_kmh = std::max(0.0f, stop_output.target_speed_kmh);
      brake_level = 0.0f;
      out.longitudinal_phase = AccLongitudinalPhase::Accelerating;
    } else if (stop_output.state == AccStopState::Stopping) {
      out.speed_hold_recommended = false;
      target_speed_kmh = 0.0f;
      out.longitudinal_phase = AccLongitudinalPhase::Braking;
    }

    out.speed_kmh = target_speed_kmh;
    out.brake_0_10 = brake_level;
    out.target_id = lead_id;
    out.target_forward_m = has_lead ? filt_dist_m : 0.0f;
    out.relative_speed_mps = has_lead ? rel_speed_mps : 0.0f;

    // 7.2 lead info + TTC (mean)
    if (has_lead) {
      out.TargetSpeedKmh = estimated_lead_speed_kmh;
      out.Targetdistance = filt_dist_m;

      const float closing_speed_mps = std::max(0.0f, -rel_speed_mps);
      if (closing_speed_mps > 0.5f) out.TargetTTC = filt_dist_m / closing_speed_mps;
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
  StopAndGoController stop_and_go_;

  std::unordered_map<int, AccKalmanAdapter> lead_filters_;

  int   last_frame_ = -1;
  int   previous_lead_id_ = -1;
  float ego_speed_est_mps_ = 0.0f;
  float last_accel_cmd_mps2_ = 0.0f;
  float cut_in_elapsed_s_ = 0.0f;
  float cut_in_start_accel_mps2_ = 0.0f;
};

} // namespace acc
