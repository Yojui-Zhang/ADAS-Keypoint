#include "VehicleSkeletonAPI.h"
#include "VehicleSkeletonProcessor.h"

namespace vehicle_skeleton {

bool RunVehicleSkeletonAndHeading(
    const cv::Mat& src_frame,
    cv::Mat& out_frame,
    std::vector<TrackingBox>& world_result,
    const SkeletonKptLayout& layout,
    const SkeletonDrawParams& draw_params,
    float ego_heading_deg
) {
    auto r = ProcessVehicleSkeletonAndHeading(src_frame, world_result, layout, draw_params, ego_heading_deg);
    out_frame = r.drawn;
    return true;
}

} // namespace vehicle_skeleton

