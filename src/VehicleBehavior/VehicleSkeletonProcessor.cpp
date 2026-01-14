#include "VehicleSkeletonProcessor.h"
#include "VehicleSkeletonHeading.h"
#include "VehicleSkeletonDrawer.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace vehicle_skeleton {

static void drawHeadingAnnotation(cv::Mat& img, const TrackingBox& tb, const SkeletonDrawParams& p) {
    if (!p.draw_heading_text) return;

    std::ostringstream oss;
    if (tb.target_heading_valid) {
        oss << "Yaw: " << std::fixed << std::setprecision(1) << tb.target_heading_deg << " deg";
    } else {
        oss << "Yaw: N/A";
    }

    cv::Point org(tb.box.x, std::max(0, tb.box.y - 5));
    cv::putText(img, oss.str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0,0,0), 3, cv::LINE_AA);
    cv::putText(img, oss.str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255,255,255), 1, cv::LINE_AA);
}

ProcessorOutput ProcessVehicleSkeletonAndHeading(
    const cv::Mat& src_frame,
    std::vector<TrackingBox>& world_result,
    const SkeletonKptLayout& layout,
    const SkeletonDrawParams& draw_params,
    float ego_heading_deg
) {
    ProcessorOutput out;
    out.drawn = src_frame.empty() ? cv::Mat() : src_frame.clone();

    for (auto& tb : world_result) {
        // 只處理車輛類別 = 1
        if (tb.class_id != 1) continue;

        // world kpts -> heading
        auto hr = ComputeVehicleHeadingFromWorldKpts(tb.kpts, layout, ego_heading_deg);
        tb.target_heading_valid = hr.valid;
        tb.target_heading_deg = hr.valid ? hr.heading_deg : std::numeric_limits<float>::quiet_NaN();

        // draw skeleton on image (needs pixel kpts from history/last)
        // if (!out.drawn.empty()) {
        //     (void)DrawVehicleSkeletonOnImage(out.drawn, tb, layout, draw_params);
        //     // optionally draw bounding box
        //     cv::rectangle(out.drawn, tb.box, draw_params.color, 2, cv::LINE_AA);
        //     drawHeadingAnnotation(out.drawn, tb, draw_params);
        // }

        out.processed_vehicle_count++;

    }

    return out;
}

} // namespace vehicle_skeleton

