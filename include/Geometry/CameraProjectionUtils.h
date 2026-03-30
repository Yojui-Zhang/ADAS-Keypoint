#pragma once

#include <opencv2/core.hpp>

class CameraModel;

cv::Point3f RawWorldPointFromVehicleMeters(float forward_m, float left_m);

cv::Point2f ProjectRawWorldPointToImage(const CameraModel& cam,
                                        const cv::Size& image_size,
                                        const cv::Point3f& raw_world_cm);

cv::Point2f ProjectVehicleGroundPointToImage(const CameraModel& cam,
                                             const cv::Size& image_size,
                                             float forward_m,
                                             float left_m);

bool IsProjectedPointInsideImage(const cv::Size& image_size,
                                 const cv::Point2f& point,
                                 float padding_px = 0.0f);
