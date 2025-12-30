#pragma once
#ifndef KEYPOINT_FILTER_SWITCH_H_
#define KEYPOINT_FILTER_SWITCH_H_

#include <vector>
#include <opencv2/core/types.hpp>

#include "KeypointEMA.h"
#include "KeypointKF.h"

namespace sort_kpt {

class KeypointFilterSwitch {
public:
    enum FilterType {
        EMA = 0,
        KF  = 1
    };

    KeypointFilterSwitch();

    // Update keypoint state.
    // - meas == nullptr: no measurement for this track in this frame (predict/hold).
    void Update(int track_id, const std::vector<cv::Point3f>* meas);

    // Retrieve last filtered keypoints for a track.
    bool GetOutput(int track_id, std::vector<cv::Point3f>& out) const;

    // Remove track state (call when a KalmanTracker is deleted).
    void Erase(int track_id);

    // Clear all state.
    void Clear();

    FilterType Active() const { return active_; }

private:
    FilterType ResolveActive_();

    FilterType active_;
    kpt_ema::KeypointEMAFilter ema_;
    kpt_kf::KeypointKFFilter  kf_;
};

// Global singleton used by SORT.
KeypointFilterSwitch& GlobalKeypointFilter();

} // namespace sort_kpt

#endif // KEYPOINT_FILTER_SWITCH_H_
