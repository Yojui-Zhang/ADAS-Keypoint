#pragma once

#include <limits>

#include "CollisionAssistApi.h"

namespace collision {

struct AebAudioGateOutput {
  bool candidate = false;
  bool active = false;
  bool rising_edge = false;

  int threat_id = -1;
  float confirm_time_s = 0.0f;
  float release_time_s = 0.0f;

  float ttc_s = std::numeric_limits<float>::infinity();
  float forward_m = std::numeric_limits<float>::infinity();
  float approach_speed_mps = 0.0f;
  int track_age_frames = 0;
  float track_score = 0.0f;
};

class AebAudioGate {
public:
  AebAudioGateOutput Update(const CollisionAssistOutput& ca,
                            const CollisionAssistConfig& cfg,
                            float dt_s);

  void Reset() noexcept;

private:
  int candidate_id_ = -1;
  float confirm_time_s_ = 0.0f;
  float release_time_s_ = 0.0f;
  bool active_ = false;
};

}  // namespace collision
