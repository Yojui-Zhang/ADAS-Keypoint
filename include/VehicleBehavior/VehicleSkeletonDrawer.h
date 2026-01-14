#pragma once
#include "config.h"
#include "VehicleSkeletonTypes.h"
#include <opencv2/core.hpp>

namespace vehicle_skeleton {

bool DrawVehicleSkeletonOnImage(
    cv::Mat& img,
    const TrackingBox& tb,
    const SkeletonKptLayout& layout,
    const SkeletonDrawParams& draw_params
);

} // namespace vehicle_skeleton

