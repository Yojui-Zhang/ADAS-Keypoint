#include "GroundPlane.h"
#include <cmath>

GroundPlane GroundPlane::Z0()
{
    return GroundPlane(cv::Vec3f(0.f, 0.f, 1.f), 0.f);
}

GroundPlane::GroundPlane(const cv::Vec3f& n_world, float d_world)
: n_(n_world), d_(d_world)
{
    // Normalize normal for numerical stability
    float norm = std::sqrt(n_[0]*n_[0] + n_[1]*n_[1] + n_[2]*n_[2]);
    if (norm > 1e-8f) {
        n_ *= (1.0f / norm);
        d_ *= (1.0f / norm);
    }
}
