#pragma once
#ifndef KEYPOINT_FILTER_SWITCH_H_
#define KEYPOINT_FILTER_SWITCH_H_

#include <vector>
#include <opencv2/core/types.hpp>

#include "KeypointEMA.h"
#include "KeypointKF.h"

namespace sort_kpt {

struct KeypointFilterConfig {
    // 濾波器型態：-1=依編譯/環境變數決定，0=EMA，1=KF
    int filter_type = -1;
    // 是否允許環境變數覆寫 filter_type
    bool allow_env_override = true;
    // EMA 濾波參數
    kpt_ema::KeypointEMAFilter::Params ema_params{};
    // KF 濾波參數
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
    FilterType active_;                 // 當前啟用的濾波器
    kpt_ema::KeypointEMAFilter ema_;
    kpt_kf::KeypointKFFilter  kf_;
};

// Global singleton used by SORT.
KeypointFilterSwitch& GlobalKeypointFilter();
void ConfigureGlobalKeypointFilter(const KeypointFilterConfig& cfg);

} // namespace sort_kpt

#endif // KEYPOINT_FILTER_SWITCH_H_
