///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H 2

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

#define StateType Rect_<float>

// This class represents the internal state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
	KalmanTracker()
	{
		init_kf(StateType());
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_class = -1;
		m_id = kf_count;
		score = 0.0f;
		// kf_count++;  // 你的原本邏輯保留（若你確定 default ctor 不會被用到就 OK）
	}

	KalmanTracker(StateType initRect, float initScore)
	{
		init_kf(initRect);
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_id = kf_count;
		m_class = -1;
		score = initScore;
		kf_count++;
	}

	~KalmanTracker()
	{
		m_history.clear();
	}

	StateType predict();

	// ✅ 主介面：外部一般傳 Rect(x,y,w,h) + score
	void update(StateType stateMat, float score);

	// ✅ 兼容你目前 kalman.cc 寫法：若別處是 update(x,y,w,h,score)，不用改
	inline void update(float x, float y, float w, float h, float score) {
		update(StateType(x, y, w, h), score);
	}

	StateType get_state();
	StateType get_rect_xysr(float cx, float cy, float s, float r);

	static int kf_count;

	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;
	int m_class;
	float score;

private:
	void init_kf(StateType stateMat);

	cv::KalmanFilter kf;
	cv::Mat measurement;

	std::vector<StateType> m_history;
};

#endif
