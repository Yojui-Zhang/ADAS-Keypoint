#pragma once
#ifndef KEYPOINT_FILTER_UTILS_H_
#define KEYPOINT_FILTER_UTILS_H_

#include <algorithm>
#include <cmath>

namespace kpt_utils {

// Safe clamp utility (C++11 compatible).
template <typename T>
inline T clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

// Normalize keypoint confidence/visibility to [0,1].
// Supports:
//   - [0,1] convention
//   - [0,100] convention
inline float normalize_conf(float conf_raw) {
    float conf = conf_raw;
    // Heuristic: values > 1.5 are likely in [0,100].
    if (conf > 1.5f) conf *= 0.01f;
    return clamp(conf, 0.0f, 1.0f);
}

// Basic measurement validity check: positive coords and finite.
inline bool meas_xy_valid(float x, float y) {
    return (x > 0.0f) && (y > 0.0f) && std::isfinite(x) && std::isfinite(y);
}

} // namespace kpt_utils

#endif // KEYPOINT_FILTER_UTILS_H_
