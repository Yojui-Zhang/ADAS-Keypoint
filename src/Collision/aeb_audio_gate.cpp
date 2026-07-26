#include "aeb_audio_gate.h"

#include <algorithm>
#include <cmath>

namespace collision {

AebAudioGateOutput AebAudioGate::Update(const CollisionAssistOutput& ca,
                                        const CollisionAssistConfig& cfg,
                                        float dt_s,
                                        float ego_speed_kmh) {
  dt_s = std::clamp(std::isfinite(dt_s) ? dt_s : 0.05f, 0.005f, 0.20f);
  ego_speed_kmh = std::isfinite(ego_speed_kmh)
                      ? std::max(0.0f, ego_speed_kmh)
                      : 0.0f;

  const bool low_speed = ego_speed_kmh < 15.0f;
  const float confirm_time_s =
      low_speed ? cfg.aeb_audio_low_speed_confirm_time_s
                : cfg.aeb_audio_confirm_time_s;
  const float immediate_ttc_s =
      low_speed ? cfg.aeb_audio_low_speed_immediate_ttc_s
                : cfg.aeb_audio_immediate_ttc_s;
  const float immediate_forward_m =
      low_speed ? cfg.aeb_audio_low_speed_immediate_forward_m
                : cfg.aeb_audio_immediate_forward_m;

  const float forward_m = ca.threat_traj.p0.x;
  const float longitudinal_closing_mps =
      std::max(0.0f, -ca.threat_vrel.x);
  const bool current_path_close =
      std::isfinite(ca.threat_dist_now_m) &&
      ca.threat_dist_now_m <=
          std::max(0.0f, cfg.aeb_audio_current_path_half_width_m);
  const bool longitudinal_closing_fast =
      longitudinal_closing_mps >=
      std::max(0.0f, cfg.aeb_audio_min_longitudinal_closing_mps);
  const bool valid_threat =
      ca.collision_warning_raw &&
      ca.threat_id >= 0 &&
      std::isfinite(ca.threat_ttc_s) &&
      ca.threat_ttc_s > 0.0f &&
      std::isfinite(forward_m) &&
      forward_m > 0.0f;

  const bool mature_track =
      ca.threat_traj.age_frames >=
      std::max(0, cfg.aeb_audio_min_track_age_frames);

  const bool reliable_detection =
      ca.threat_traj.detection_score >=
      std::max(0.0f, cfg.aeb_audio_min_track_score);

  const bool meaningful_approach =
      ca.threat_approach_speed_mps >=
      std::max(0.0f, cfg.aeb_audio_min_approach_speed_mps);

  const bool ttc_danger =
      ca.threat_ttc_s <=
      std::max(0.0f, cfg.aeb_warning_ttc_s);

  const bool candidate =
      valid_threat &&
      mature_track &&
      reliable_detection &&
      meaningful_approach &&
      ttc_danger &&
      current_path_close &&
      longitudinal_closing_fast;

  if (candidate) {
    if (candidate_id_ == ca.threat_id) {
      confirm_time_s_ += dt_s;
    } else {
      candidate_id_ = ca.threat_id;
      confirm_time_s_ = dt_s;
    }
    release_time_s_ = 0.0f;
  } else {
    confirm_time_s_ = 0.0f;
    candidate_id_ = -1;

    if (active_) {
      release_time_s_ += dt_s;
    }
  }

  const bool immediate_danger =
      valid_threat &&
      mature_track &&
      reliable_detection &&
      meaningful_approach &&
      current_path_close &&
      longitudinal_closing_fast &&
      ca.threat_ttc_s <= std::max(0.0f, immediate_ttc_s) &&
      forward_m <= std::max(0.0f, immediate_forward_m);

  const bool confirmed_danger =
      candidate &&
      confirm_time_s_ >=
          std::max(0.0f, confirm_time_s);

  if (!valid_threat ||
      ca.threat_ttc_s >= std::max(0.0f, cfg.aeb_audio_ttc_off_s)) {
    same_threat_safe_time_s_ += dt_s;
  } else {
    same_threat_safe_time_s_ = 0.0f;
  }

  if (same_threat_safe_time_s_ >=
      std::max(0.0f, cfg.aeb_audio_same_threat_rearm_safe_s)) {
    same_threat_armed_ = true;
  }

  const bool trigger_allowed =
      ca.threat_id != last_triggered_threat_id_ ||
      same_threat_armed_;

  const bool previous_active = active_;

  const char* trigger_reason = "none";
  if (!active_ && trigger_allowed && (confirmed_danger || immediate_danger)) {
    active_ = true;
    trigger_reason = immediate_danger ? "immediate" : "confirmed";
  } else if (!trigger_allowed && (confirmed_danger || immediate_danger)) {
    trigger_reason = "same_threat_suppressed";
  } else if (candidate) {
    trigger_reason = "candidate";
  }

  const bool clearly_safe =
      !valid_threat ||
      ca.threat_ttc_s >=
          std::max(0.0f, cfg.aeb_audio_ttc_off_s);

  if (active_ &&
      clearly_safe &&
      release_time_s_ >=
          std::max(0.0f, cfg.aeb_audio_release_time_s)) {
    active_ = false;
  }

  AebAudioGateOutput output;
  output.candidate = candidate;
  output.active = active_;
  output.rising_edge = !previous_active && active_;
  if (output.rising_edge) {
    last_triggered_threat_id_ = ca.threat_id;
    same_threat_safe_time_s_ = 0.0f;
    same_threat_armed_ = false;
  }
  output.threat_id = valid_threat ? ca.threat_id : -1;
  output.confirm_time_s = confirm_time_s_;
  output.release_time_s = release_time_s_;
  output.trigger_reason = trigger_reason;
  output.ttc_s = valid_threat ? ca.threat_ttc_s
                              : std::numeric_limits<float>::infinity();
  output.forward_m = valid_threat ? forward_m
                                  : std::numeric_limits<float>::infinity();
  output.approach_speed_mps =
      valid_threat ? ca.threat_approach_speed_mps : 0.0f;
  output.track_age_frames = valid_threat ? ca.threat_traj.age_frames : 0;
  output.track_score = valid_threat ? ca.threat_traj.detection_score : 0.0f;
  output.longitudinal_closing_mps =
      valid_threat ? longitudinal_closing_mps : 0.0f;
  output.current_path_distance_m =
      valid_threat ? ca.threat_dist_now_m
                   : std::numeric_limits<float>::infinity();
  output.same_threat_armed = same_threat_armed_;

  return output;
}

void AebAudioGate::Reset() noexcept {
  candidate_id_ = -1;
  confirm_time_s_ = 0.0f;
  release_time_s_ = 0.0f;
  active_ = false;
  last_triggered_threat_id_ = -1;
  same_threat_safe_time_s_ = 0.0f;
  same_threat_armed_ = true;
}

}  // namespace collision
