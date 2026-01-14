#pragma once
#include "config.h"
#include "VehicleSkeletonTypes.h"
#include <opencv2/core.hpp>
#include <vector>

namespace vehicle_skeleton {

// 你在 main() 只需要 include 這個頭檔，呼叫這個函式即可
// - input: src_frame + WorldResult
// - output: out_frame (畫好骨架) + WorldResult(寫回角度)
bool RunVehicleSkeletonAndHeading(
    const cv::Mat& src_frame,
    cv::Mat& out_frame,
    std::vector<TrackingBox>& world_result,
    const SkeletonKptLayout& layout,
    const SkeletonDrawParams& draw_params = SkeletonDrawParams{},
    float ego_heading_deg = 0.0f
);

} // namespace vehicle_skeleton

