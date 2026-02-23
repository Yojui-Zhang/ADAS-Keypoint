#pragma once
#include <opencv2/core.hpp>
#include "SortTracking.h"
#include "KeypointFilterSwitch.h"

bool Classify_and_icon_init(const char* classify_model_path, const char* Icon_path);

void tflite_set_sort_config(const SORTTRACKING::SortTrackingConfig& sort_cfg,
                            const sort_kpt::KeypointFilterConfig& kpt_cfg);

bool tflite_init(const char* lanepose_model_path, const cv::Mat& first_frame);
std::vector<TrackingBox> tflite_run_frame(const cv::Mat& frame,
                                                cv::Mat& out_bgr,
                                          int classify_model_width,
                                          int classify_model_height);
