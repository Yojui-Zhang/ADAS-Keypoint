#pragma once
#include <opencv2/core.hpp>


bool Classify_and_icon_init(const char* classify_model_path, const char* Icon_path);

bool tflite_init(const char* lanepose_model_path, const cv::Mat& first_frame);
bool tflite_run_frame(const cv::Mat& frame,
                      cv::Mat& out_bgr,
                      int classify_model_width,
                      int classify_model_height);
