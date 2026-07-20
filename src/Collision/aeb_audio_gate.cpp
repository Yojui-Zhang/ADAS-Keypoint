#include "aeb_audio_gate.h"

#include <algorithm>
#include <cmath>

namespace collision {

AebAudioGateOutput AebAudioGate::Update(const CollisionAssistOutput& ca,
                                        const CollisionAssistConfig& cfg,
                                        float dt_s) {
  dt_s = std::clamp(std::isfinite(dt_s) ? dt_s : 0.05f, 0.005f, 0.20f);

  const float forward_m = ca.threat_traj.p0.x;
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
      ttc_danger;

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
      (ca.threat_ttc_s <=
           std::max(0.0f, cfg.aeb_audio_immediate_ttc_s) ||
       forward_m <=
           std::max(0.0f, cfg.aeb_audio_immediate_forward_m));

  const bool confirmed_danger =
      candidate &&
      confirm_time_s_ >=
          std::max(0.0f, cfg.aeb_audio_confirm_time_s);

  const bool previous_active = active_;

  if (!active_ && (confirmed_danger || immediate_danger)) {
    active_ = true;
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
  output.threat_id = valid_threat ? ca.threat_id : -1;
  output.confirm_time_s = confirm_time_s_;
  output.release_time_s = release_time_s_;
  output.ttc_s = valid_threat ? ca.threat_ttc_s
                              : std::numeric_limits<float>::infinity();
  output.forward_m = valid_threat ? forward_m
                                  : std::numeric_limits<float>::infinity();
  output.approach_speed_mps =
      valid_threat ? ca.threat_approach_speed_mps : 0.0f;
  output.track_age_frames = valid_threat ? ca.threat_traj.age_frames : 0;
  output.track_score = valid_threat ? ca.threat_traj.detection_score : 0.0f;

  return output;
}

void AebAudioGate::Reset() noexcept {
  candidate_id_ = -1;
  confirm_time_s_ = 0.0f;
  release_time_s_ = 0.0f;
  active_ = false;
}

}  // namespace collision
