#include "KeypointEMA.h"

#include <algorithm>

#include "KeypointFilterUtils.h"

namespace kpt_ema {

KeypointEMAFilter::KeypointEMAFilter() : KeypointEMAFilter(Params()) {}

KeypointEMAFilter::KeypointEMAFilter(const Params& params) : params_(params) {}

float KeypointEMAFilter::ConfToAlpha(float conf) const {
    conf = Clamp(conf, 0.0f, 1.0f);
    float t = (conf <= params_.conf_thr) ? 0.f : (conf - params_.conf_thr) / (1.f - params_.conf_thr);
    return params_.alpha_lo + (params_.alpha_hi - params_.alpha_lo) * t;
}

void KeypointEMAFilter::Update(int track_id, const std::vector<cv::Point3f>* meas) {
    auto& st = states_[track_id];

    // Init / resize if keypoint count changes.
    if (meas && static_cast<int>(meas->size()) != st.num_kpts) {
        st.kpts.assign(meas->size(), KptEMA{});
        st.num_kpts = static_cast<int>(meas->size());
    } else if (!meas && st.num_kpts == 0) {
        // No measurement and no state: nothing to output.
        outputs_.erase(track_id);
        return;
    }

    std::vector<cv::Point3f> out;
    out.resize(st.num_kpts, cv::Point3f(0.f, 0.f, 0.f));

    if (meas) {
        st.missed = 0;
        for (int i = 0; i < st.num_kpts; ++i) {
            const float mx = (*meas)[i].x;
            const float my = (*meas)[i].y;
            const float mc = (*meas)[i].z; // keypoint confidence

            // Support both [0,1] and [0,100] confidence conventions.
            const float conf = kpt_utils::normalize_conf(mc);

            // Update only for valid measurements.
            if (conf > 0.f && mx > 0.f && my > 0.f) {
                const float alpha = ConfToAlpha(conf);
                if (!st.kpts[i].initialized) {
                    st.kpts[i].pt = cv::Point2f(mx, my);
                    st.kpts[i].initialized = true;
                } else {
                    st.kpts[i].pt.x = alpha * mx + (1.f - alpha) * st.kpts[i].pt.x;
                    st.kpts[i].pt.y = alpha * my + (1.f - alpha) * st.kpts[i].pt.y;
                }
                out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, mc);
            } else {
                // Invalid measurement: keep last (if initialized), set z=0.
                if (st.kpts[i].initialized) {
                    out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, 0.f);
                }
            }
        }
    } else {
        // No measurement: keep last (z=0), missed++.
        st.missed += 1;
        for (int i = 0; i < st.num_kpts; ++i) {
            if (st.kpts[i].initialized) {
                out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, 0.f);
            }
        }
    }

    outputs_[track_id] = std::move(out);
}

bool KeypointEMAFilter::GetOutput(int track_id, std::vector<cv::Point3f>& out) const {
    auto it = outputs_.find(track_id);
    if (it == outputs_.end()) return false;
    out = it->second;
    return true;
}

void KeypointEMAFilter::Erase(int track_id) {
    states_.erase(track_id);
    outputs_.erase(track_id);
}

void KeypointEMAFilter::Clear() {
    states_.clear();
    outputs_.clear();
}

} // namespace kpt_ema