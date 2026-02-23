#pragma once
#ifndef SORTTRACKING_H_
#define SORTTRACKING_H_
#include "Hungarian.h"
#include "kalman.h"
#include "config.h"

#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h> // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <unistd.h>
#include <set>


class SORTTRACKING {

public:

    struct SortTrackingConfig {
        int max_age = 5;
        int min_hits = 3;
        double iou_threshold = 0.3;
        int history_length = 10;
    };

    // global variables for counting
    #define CNUM 20

    SORTTRACKING();
    explicit SORTTRACKING(const SortTrackingConfig& cfg);
    ~SORTTRACKING();
    std::vector<TrackingBox> TrackingResult(const std::vector<Object> &bboxes);

    void SetConfig(const SortTrackingConfig& cfg);
    SortTrackingConfig GetConfig() const { return config_; }

    std::vector<KalmanTracker> trackers;
private:

    int frame_count = 0;

    SortTrackingConfig config_;
	
	
	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;

	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;
    std::vector<TrackingBox> detData;
    cv::Mat h=Mat::ones(3,3,CV_64FC1);
    cv::Mat m=Mat::ones(3,1,CV_64FC1);
    double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);
    double vehicle_dis(double x_data, double y_data, double z_data);

    // 歷史紀錄暫存區：key=ID, value=該 ID 的歷史資料佇列
    std::map<int, std::deque<TrackingBox>> trajectory_registry; 

    // 新增功能函式宣告
    void UpdateTrajectory(std::vector<TrackingBox>& results, int history_len);

};

#endif // !MOBILEFACENET_H_
