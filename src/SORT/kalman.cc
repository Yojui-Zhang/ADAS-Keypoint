///////////////////////////////////////////////////////////////////////////////
// kalman.cc: KalmanTracker Class Implementation

#include "kalman.h"
#include <cmath>
#include <algorithm>

int KalmanTracker::kf_count = 0;

static inline float clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(x, hi));
}

// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum   = 7;
	int measureNum = 4;
	kf = KalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);

	// 初始不確定度
	setIdentity(kf.errorCovPost, Scalar::all(1));

	// 過程噪聲
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));

	// baseline 量測噪聲（score=1 時大致使用這個級距）
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-4));

	// ✅ 避免 stateMat 為空導致 0/0 -> NaN
	float w = std::max(1e-3f, stateMat.width);
	float h = std::max(1e-3f, stateMat.height);

	// init state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + w * 0.5f;
	kf.statePost.at<float>(1, 0) = stateMat.y + h * 0.5f;
	kf.statePost.at<float>(2, 0) = w * h;         // s = area
	kf.statePost.at<float>(3, 0) = w / h;         // r = aspect ratio
}

// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	Mat p = kf.predict();
	m_age += 1;

	if (m_time_since_update > 0) m_hit_streak = 0;
	m_time_since_update += 1;

	StateType predictBox = get_rect_xysr(p.at<float>(0, 0),
	                                    p.at<float>(1, 0),
	                                    p.at<float>(2, 0),
	                                    p.at<float>(3, 0));

	m_history.push_back(predictBox);
	return m_history.back();
}

// ✅ 正確版：update(Rect(x,y,w,h), score) -> measurement=[cx,cy,s,r] 並做 score→R
void KalmanTracker::update(StateType stateMat, float det_score)
{
	// SORT bookkeeping（這些你原本 kalman.cc 沒做，會讓追蹤狀態壞掉）
	m_time_since_update = 0;
	m_history.clear();
	m_hits += 1;
	m_hit_streak += 1;

	score = det_score;

	// Rect(x,y,w,h) -> [cx,cy,s,r]
	const float w = std::max(1e-3f, stateMat.width);
	const float h = std::max(1e-3f, stateMat.height);

	const float cx = stateMat.x + w * 0.5f;
	const float cy = stateMat.y + h * 0.5f;
	const float s  = w * h;
	const float r  = w / h;

	measurement.at<float>(0, 0) = cx;
	measurement.at<float>(1, 0) = cy;
	measurement.at<float>(2, 0) = s;
	measurement.at<float>(3, 0) = r;

	// ========= score-driven measurement noise R =========
	// paper: R = R0/(score^2 + eps)
	const float sscore = clampf(det_score, 0.0f, 1.0f);
	const float eps    = 1e-3f;

	// baseline：與 init_kf 的 measurementNoiseCov 同級距
	const float R0     = 1e-4f;

	float r_meas = R0 / (sscore * sscore + eps);

	// clamp：避免 score 太小導致 R 爆炸、濾波完全不跟量測
	// 你可以依資料再調整 r_max（常見 1e-3 ~ 1e-1）
	const float r_min = 1e-6f;
	const float r_max = 1e-2f;

	r_meas = clampf(r_meas, r_min, r_max);

	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(r_meas));
	// ================================================

	// correct
	kf.correct(measurement);


	// printf("id=%d score=%.2f r_meas=%g\n", m_id, det_score, r_meas);

}

// Return the current state vector
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}

// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = std::sqrt(std::max(1e-12f, s * r));
	float h = std::max(1e-6f, s / std::max(1e-6f, w));
	float x = (cx - w * 0.5f);
	float y = (cy - h * 0.5f);

	if (x < 0 && cx > 0) x = 0;
	if (y < 0 && cy > 0) y = 0;

	return StateType(x, y, w, h);
}
