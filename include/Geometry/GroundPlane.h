#pragma once
#include <opencv2/core.hpp>

// Plane in world frame: n^T * X + d = 0
// n is unit normal recommended.
class GroundPlane
{
public:
    GroundPlane() = default;

    // Default: Z=0 plane (n=[0,0,1], d=0)
    static GroundPlane Z0();

    // Construct by normal and d
    GroundPlane(const cv::Vec3f& n_world, float d_world);

    const cv::Vec3f& n() const { return n_; }
    float d() const { return d_; }

private:
    cv::Vec3f n_{0.f, 0.f, 1.f};
    float d_{0.f};
};
