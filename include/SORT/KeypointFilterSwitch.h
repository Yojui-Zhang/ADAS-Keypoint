#pragma once
#ifndef KEYPOINT_FILTER_SWITCH_H_
#define KEYPOINT_FILTER_SWITCH_H_

#include <vector>
#include <opencv2/core/types.hpp>

#include "KeypointEMA.h"
#include "KeypointKF.h"

namespace sort_kpt {

struct KeypointFilterConfig {
    // -1: use compile-time/env default, 0: EMA, 1: KF
    int filter_type = -1;
    bool allow_env_override = true;
    kpt_ema::KeypointEMAFilter::Params ema_params{};
    kpt_kf::KeypointKFFilter::Params kf_params{};
};

class KeypointFilterSwitch {
public:
    enum FilterType {
        EMA = 0,
        KF  = 1
    };

    explicit KeypointFilterSwitch(const KeypointFilterConfig& cfg = KeypointFilterConfig{});

    void SetConfig(const KeypointFilterConfig& cfg);
    KeypointFilterConfig GetConfig() const { return cfg_; }

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
    FilterType ResolveActive_(const KeypointFilterConfig& cfg) const;

    KeypointFilterConfig cfg_;
    FilterType active_;
    kpt_ema::KeypointEMAFilter ema_;
    kpt_kf::KeypointKFFilter  kf_;
};

// Global singleton used by SORT.
KeypointFilterSwitch& GlobalKeypointFilter();
void ConfigureGlobalKeypointFilter(const KeypointFilterConfig& cfg);

} // namespace sort_kpt

#endif // KEYPOINT_FILTER_SWITCH_H_
