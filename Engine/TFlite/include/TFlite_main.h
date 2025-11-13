#pragma once
#include <opencv2/core.hpp>


bool Classify_and_icon_init(const char* classify_model_path, const char* Icon_path);

// 初始化 TFLite（載入模型、根據第一張 frame 計算縮放）
bool tflite_init(const char* lanepose_model_path, const cv::Mat& first_frame);

// 跑單張影像：內含 preprocess → Invoke → 後處理與畫圖
// 輸出結果放到 out_bgr（BGR Mat）
bool tflite_run_frame(const cv::Mat& frame,
                      cv::Mat& out_bgr,
                      int classify_model_width,
                      int classify_model_height);
