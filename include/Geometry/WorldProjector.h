#pragma once
#include <opencv2/core.hpp>
#include <limits>
#include "CameraModel.h"
#include "GroundPlane.h"

class WorldProjector
{
public:
    WorldProjector(CameraModel cam, GroundPlane plane);

    // Project pixel to world point on plane.
    // Return false when ray is parallel or intersection behind camera.
    bool pixelToWorldOnPlane(const cv::Point2f& px, cv::Point3f& out_w) const;

    static cv::Point3f NaNPoint()
    {
        float q = std::numeric_limits<float>::quiet_NaN();
        return cv::Point3f(q,q,q);
    }

private:
    CameraModel cam_;
    GroundPlane plane_;
};
