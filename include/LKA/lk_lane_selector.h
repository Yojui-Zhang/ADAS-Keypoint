#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "lane_keeping.h" // for ControlConfig
#include "SortTracking.h" // for TrackingBox

namespace lane_keeping {
namespace internal {

// 用於封裝單側車道的擬合結果
struct LaneCandidate {
    bool valid = false;             // 是否找到有效車道
    cv::Vec3d poly = {0, 0, 0};     // 擬合係數 [a2, a1, a0]
    std::vector<cv::Point2f> pts;   // 原始過濾後的點
    float abs_y_eval = 1e9f;        // 用於評估優劣的橫向距離絕對值
    std::string debug_info;         // 擬合的除錯訊息
};

// 封裝左右車道的搜尋結果
struct LanePair {
    LaneCandidate left;
    LaneCandidate right;
};

// 核心函式：從 WorldResult 中挑選出最佳的左/右車道
LanePair FindBestLaneCandidates(const std::vector<TrackingBox>& world_result,
                                const ControlConfig& cfg);

} // namespace internal
} // namespace lane_keeping
