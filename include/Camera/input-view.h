#ifndef VIDEO_INIT_H
#define VIDEO_INIT_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "config.h"

struct InputViewConfig {
    std::string video_path = "../video/1280x720/vecow-demo.mp4";  // 影片檔路徑（camera_index < 0 時使用）
    int camera_index = -1;                                        // 攝影機索引（>=0 時優先使用即時鏡頭）
    int capture_width = input_video_width;                        // 影像擷取寬度（像素）
    int capture_height = input_video_height;                      // 影像擷取高度（像素）
    std::string window_name = "Screen";                           // 顯示視窗名稱
    bool fullscreen = true;                                       // 是否全螢幕顯示
};

// 宣告初始化函式
// 傳入 cap 和 frame 的引用(Reference)，這樣函式內的修改會直接影響 main 裡面的變數
int InitInputAndDisplay(cv::VideoCapture& cap,
                        cv::Mat& frame,
                        const InputViewConfig& cfg = InputViewConfig{},
                        bool use_opengl_display = false);

#endif // VIDEO_INIT_H
