#include "KeypointKF.h"
#include <cmath>

#include "KeypointFilterUtils.h"

namespace kpt_kf {

Kpt2DKF::Kpt2DKF()
    : kf_(4, 2, 0, CV_32F),
      meas_(2, 1, CV_32F),
      inited_(false),
      process_var_(1.0f),
      last_xy_(0.f, 0.f)
{
    setupModel_();
}

void Kpt2DKF::setupModel_()
{
    // dt = 1 frame
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Will be overwritten in init()
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(process_var_));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(25.0f));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0f));

    // initial state
    kf_.statePost = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);
}

void Kpt2DKF::reset()
{
    inited_ = false;
    last_xy_ = cv::Point2f(0.f, 0.f);
    setupModel_();
}

void Kpt2DKF::init(float x, float y,
                   float init_pos_var, float init_vel_var,
                   float process_var)
{
    process_var_ = process_var;

    setupModel_();

    kf_.statePost.at<float>(0) = x;
    kf_.statePost.at<float>(1) = y;
    kf_.statePost.at<float>(2) = 0.f;
    kf_.statePost.at<float>(3) = 0.f;

    // initial covariance
    kf_.errorCovPost = cv::Mat::zeros(4, 4, CV_32F);
    kf_.errorCovPost.at<float>(0, 0) = init_pos_var;
    kf_.errorCovPost.at<float>(1, 1) = init_pos_var;
    kf_.errorCovPost.at<float>(2, 2) = init_vel_var;
    kf_.errorCovPost.at<float>(3, 3) = init_vel_var;

    // process noise
    kf_.processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
    kf_.processNoiseCov.at<float>(0, 0) = process_var_;
    kf_.processNoiseCov.at<float>(1, 1) = process_var_;
    kf_.processNoiseCov.at<float>(2, 2) = process_var_;
    kf_.processNoiseCov.at<float>(3, 3) = process_var_;

    inited_ = true;
    last_xy_ = cv::Point2f(x, y);
}

cv::Point2f Kpt2DKF::predict()
{
    if (!inited_) {
        return last_xy_;
    }
    cv::Mat pred = kf_.predict();
    last_xy_.x = pred.at<float>(0);
    last_xy_.y = pred.at<float>(1);
    return last_xy_;
}

cv::Point2f Kpt2DKF::correct(float mx, float my, float meas_var)
{
    if (!inited_) {
        // If correct() is called before init, do a safe init-like behavior
        init(mx, my, 100.f, 1000.f, process_var_);
        return last_xy_;
    }

    // measurement noise
    kf_.measurementNoiseCov = cv::Mat::zeros(2, 2, CV_32F);
    kf_.measurementNoiseCov.at<float>(0, 0) = meas_var;
    kf_.measurementNoiseCov.at<float>(1, 1) = meas_var;

    meas_.at<float>(0) = mx;
    meas_.at<float>(1) = my;

    cv::Mat est = kf_.correct(meas_);
    last_xy_.x = est.at<float>(0);
    last_xy_.y = est.at<float>(1);
    return last_xy_;
}

// ------------------------- KeypointKFFilter -------------------------

KeypointKFFilter::KeypointKFFilter()
    : params_(Params())
{
}

KeypointKFFilter::KeypointKFFilter(const Params& params)
    : params_(params)
{
}

void KeypointKFFilter::Update(int track_id, const std::vector<cv::Point3f>* kpts)
{
    // If track doesn't exist and no measurement, nothing to do.
    if (!kpts && tracks_.find(track_id) == tracks_.end()) {
        return;
    }

    TrackState& st = tracks_[track_id];

    if (kpts) {
        const size_t n = kpts->size();
        if (st.kfs.size() != n) {
            st.kfs.clear();
            st.kfs.resize(n);
            st.last_out.clear();
            st.last_out.resize(n, cv::Point3f(0.f, 0.f, 0.f));
            st.initialized = false;
        }

        for (size_t i = 0; i < n; ++i) {
            const float mx = (*kpts)[i].x;
            const float my = (*kpts)[i].y;
            const float v_raw = (*kpts)[i].z; // confidence/visibility (may be 0-1 or 0-100)
            const float v = kpt_utils::normalize_conf(v_raw);

            const bool meas_valid = (v >= params_.conf_thr) && kpt_utils::meas_xy_valid(mx, my);

            // If not initialized, only initialize on a valid measurement.
            if (!st.kfs[i].isInitialized()) {
                if (meas_valid) {
                    st.kfs[i].init(mx, my, params_.init_pos_var, params_.init_vel_var, params_.process_var);
                    st.last_out[i] = cv::Point3f(mx, my, v_raw);
                } else {
                    st.last_out[i] = cv::Point3f(0.f, 0.f, 0.f);
                }
                continue;
            }

            // Predict every frame once initialized.
            const cv::Point2f pred_xy = st.kfs[i].predict();

            if (!meas_valid) {
                // Low confidence / missing -> predict-only output
                st.last_out[i] = cv::Point3f(pred_xy.x, pred_xy.y, 0.f);
                continue;
            }

            // Gating: reject outliers far away from prediction
            const float dx = mx - pred_xy.x;
            const float dy = my - pred_xy.y;
            const float dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= params_.gate_dist_px) {
                const float eps = 0.05f;
                const float vv = (v > eps) ? v : eps;
                const float meas_var = params_.meas_var_base / vv;

                cv::Point2f est_xy = st.kfs[i].correct(mx, my, meas_var);
                st.last_out[i] = cv::Point3f(est_xy.x, est_xy.y, v_raw);
            } else {
                // Outlier -> predict-only output, mark as missing this frame
                st.last_out[i] = cv::Point3f(pred_xy.x, pred_xy.y, 0.f);
            }
        }

        st.initialized = true;
        return;
    }

    // Predict-only update (no measurements this frame)
    if (!st.initialized) {
        return;
    }

    for (size_t i = 0; i < st.kfs.size(); ++i) {
        if (st.kfs[i].isInitialized()) {
            cv::Point2f pred_xy = st.kfs[i].predict();
            st.last_out[i].x = pred_xy.x;
            st.last_out[i].y = pred_xy.y;
            st.last_out[i].z = 0.f; // no measurement
        } else {
            // keep as-is
            st.last_out[i].z = 0.f;
        }
    }
}

bool KeypointKFFilter::GetOutput(int track_id, std::vector<cv::Point3f>& out) const
{
    std::map<int, TrackState>::const_iterator it = tracks_.find(track_id);
    if (it == tracks_.end()) return false;
    out = it->second.last_out;
    return true;
}

void KeypointKFFilter::Erase(int track_id)
{
    tracks_.erase(track_id);
}

void KeypointKFFilter::Clear()
{
    tracks_.clear();
}

} // namespace kpt_kf
