#ifndef VIDEO_INIT_H
#define VIDEO_INIT_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "config.h"

struct InputViewConfig {
    std::string video_path = "../video/1280x720/vecow-demo.mp4";
    int camera_index = -1;  // <0 means use video_path
    int capture_width = input_video_width;
    int capture_height = input_video_height;
    std::string window_name = "Screen";
    bool fullscreen = true;
};

// 宣告初始化函式
// 傳入 cap 和 frame 的引用(Reference)，這樣函式內的修改會直接影響 main 裡面的變數
int InitInputAndDisplay(cv::VideoCapture& cap, cv::Mat& frame, const InputViewConfig& cfg = InputViewConfig{});

#endif // VIDEO_INIT_H
