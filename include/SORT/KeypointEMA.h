#pragma once
#ifndef KEYPOINT_EMA_H_
#define KEYPOINT_EMA_H_

#include <unordered_map>
#include <vector>

#include <algorithm>  // std::min/std::max

#include <opencv2/core/types.hpp>

namespace kpt_ema {

/**
 * @brief Keypoint exponential moving average (EMA) smoother keyed by track_id.
 *
 * Behavior:
 *  - Per-track, per-keypoint EMA on (x, y).
 *  - Adaptive alpha based on keypoint confidence z (0..1):
 *      higher confidence -> larger alpha (more responsive)
 *      lower confidence  -> smaller alpha (more smoothing)
 *  - If meas == nullptr (unmatched track this frame):
 *      output last smoothed (x, y) and set z=0.
 */
class KeypointEMAFilter {
public:
    struct Params {
        float conf_thr;
        float alpha_lo;
        float alpha_hi;

        /*
        -------------------------------------------------------------------
        |   conf_thr 調高（更嚴格）
        |       更多 keypoints 被視為「低可信」
        |       EMA 更常使用偏小 alpha（靠近 alpha_lo）
        |       畫面結果：更穩、更不抖，但反應更慢、延遲更明顯（轉頭/手部快速動作會跟不上）
        |   conf_thr 調低（更寬鬆）
        |       更多 keypoints 被視為「高可信」
        |       EMA 更常使用偏大 alpha（靠近 alpha_hi）
        |------------------------------------------------------------------
        |   alpha_lo：最平滑（反應最慢）
        |       alpha_lo 調小（例如 0.05 → 0.1）
        |           低 conf 時非常平滑
        |       alpha_lo 調大（例如 0.15 → 0.3）
        |           低 conf 時也會更快跟隨新量測
        |------------------------------------------------------------------
        |   alpha_hi：最靈敏（反應最快）
        |       alpha_hi 調大（例如 0.65 → 0.85）
        |           高 conf 時幾乎貼著量測走
        |       alpha_hi 調小（例如 0.65 → 0.4）
        |           高 conf 時也保持一定程度平滑
        -------------------------------------------------------------------
        */        
        Params( float conf_thr_ = 0.2f,
                float alpha_lo_ = 0.15f,
                float alpha_hi_ = 0.65f)
            : conf_thr(conf_thr_), alpha_lo(alpha_lo_), alpha_hi(alpha_hi_) {}
    };

    // 提供一個「無參數」預設建構子，避免 default argument 觸發編譯器問題
    KeypointEMAFilter();
    explicit KeypointEMAFilter(const Params& params);

    // Update internal EMA with this frame's measurement.
    // - meas == nullptr: no observation this frame.
    void Update(int track_id, const std::vector<cv::Point3f>* meas);

    // Retrieve last output for this track_id.
    // Returns true if found.
    bool GetOutput(int track_id, std::vector<cv::Point3f>& out) const;

    // Remove all state for a track (call when tracker is deleted).
    void Erase(int track_id);

    // Clear all tracks.
    void Clear();

private:
    struct KptEMA {
        cv::Point2f pt{0.f, 0.f};
        bool initialized{false};
    };

    struct TrackState {
        std::vector<KptEMA> kpts;
        int num_kpts{0};
        int missed{0};
    };

    template <typename T>
    static inline T Clamp(T v, T lo, T hi) {
        return std::max(lo, std::min(hi, v));
    }

    float ConfToAlpha(float conf) const;

    Params params_;
    std::unordered_map<int, TrackState> states_;
    std::unordered_map<int, std::vector<cv::Point3f>> outputs_;
};

} // namespace kpt_ema

#endif // KEYPOINT_EMA_H_