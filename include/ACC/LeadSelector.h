#pragma once
#include "AccConfig.h"
#include <vector>

namespace acc {

struct LeadCandidate {
  int id = -1;
  float forward_m = 0.0f;
  float lateral_m = 0.0f;
  float score = 0.0f;     // NEW
};

class LeadSelector {
public:
  void Reset() { last_target_id_ = -1; }

  // 回傳 index（在 candidates 中），若無則 -1
  int Select(const std::vector<LeadCandidate>& candidates, const AccConfig& cfg);

private:
  int last_target_id_ = -1;
};

} // namespace acc

