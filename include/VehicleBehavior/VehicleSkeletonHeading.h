#pragma once
#include "config.h"
#include "VehicleSkeletonTypes.h"
#include <opencv2/core.hpp>

namespace vehicle_skeleton {

struct HeadingResult {
    bool valid = false;
    float heading_deg = 0.0f;     // relative to ego forward (+X) as 0 deg
    cv::Point2f front_center_w;   // (x forward, y left)
    cv::Point2f rear_center_w;
};

// world_kpts：使用 GeometryFunction.cpp 轉換後的 world kpts
// 座標語意：x=前進[m], y=左[m]
HeadingResult ComputeVehicleHeadingFromWorldKpts(
    const std::vector<cv::Point3f>& world_kpts,
    const SkeletonKptLayout& layout,
    float ego_heading_deg = 0.0f  // 若你未來要扣掉 ego yaw，可用；目前依需求 0 即可
);

} // namespace vehicle_skeleton

