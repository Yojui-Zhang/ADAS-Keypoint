#include "LeadSelector.h"
#include <limits>
#include <cmath>

namespace acc {

int LeadSelector::Select(const std::vector<LeadCandidate>& candidates, const AccConfig& cfg) {
  if (candidates.empty()) {
    last_target_id_ = -1;
    return -1;
  }

  // 先找最近（forward最小）
  int best_idx = -1;
  float best_forward = std::numeric_limits<float>::infinity();

  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    const auto& c = candidates[i];
    if (c.forward_m < best_forward) {
      best_forward = c.forward_m;
      best_idx = i;
    }
  }

  if (best_idx < 0) {
    last_target_id_ = -1;
    return -1;
  }

  // 遲滯：如果上一個目標仍存在且距離沒有明顯更差，就維持，避免目標跳動
  if (last_target_id_ >= 0) {
    int last_idx = -1;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
      if (candidates[i].id == last_target_id_) { last_idx = i; break; }
    }

    if (last_idx >= 0) {
      const float last_forward = candidates[last_idx].forward_m;
      if (last_forward <= best_forward + cfg.lead_hysteresis_m) {
        return last_idx;
      }
    }
  }

  last_target_id_ = candidates[best_idx].id;
  return best_idx;
}

} // namespace acc

