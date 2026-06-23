#include "lka_projection.h"

#include "CameraProjectionUtils.h"

namespace adas_app {

bool ProjectVehicleGroundPointToPixel(const CameraModel& cam,
                                      const cv::Mat& image,
                                      const LkaReferencePoint& point,
                                      cv::Point2f* out_pixel) {
  if (out_pixel == nullptr || point.valid == false) {
    return false;
  }

  const cv::Point2f pixel =
      ProjectVehicleGroundPointToImage(cam, image.size(), point.x_m, point.y_m);
  *out_pixel = pixel;
  return IsProjectedPointInsideImage(image.size(), pixel, 0.0f);
}

}  // namespace adas_app
