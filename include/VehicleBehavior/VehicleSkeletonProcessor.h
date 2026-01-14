#pragma once
#include "config.h"
#include "VehicleSkeletonTypes.h"
#include <opencv2/core.hpp>
#include <vector>

namespace vehicle_skeleton {

struct ProcessorOutput {
    cv::Mat drawn;
    int processed_vehicle_count = 0;
};

ProcessorOutput ProcessVehicleSkeletonAndHeading(
    const cv::Mat& src_frame,
    std::vector<TrackingBox>& world_result,      // in-place 更新 target_heading_deg
    const SkeletonKptLayout& layout,
    const SkeletonDrawParams& draw_params,
    float ego_heading_deg = 0.0f                 // 依需求：以當前車輛行使方向為 0 deg
);

} // namespace vehicle_skeleton

