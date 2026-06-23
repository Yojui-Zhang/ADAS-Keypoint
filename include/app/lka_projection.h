#pragma once

#include <opencv2/core.hpp>

#include "CameraModel.h"
#include "lane_keeping.h"

namespace adas_app {

bool ProjectVehicleGroundPointToPixel(const CameraModel& cam,
                                      const cv::Mat& image,
                                      const LkaReferencePoint& point,
                                      cv::Point2f* out_pixel);

}  // namespace adas_app
