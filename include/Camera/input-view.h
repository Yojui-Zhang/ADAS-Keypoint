#ifndef VIDEO_INIT_H
#define VIDEO_INIT_H

#include <opencv2/opencv.hpp>
#include <iostream>

// 宣告初始化函式
// 傳入 cap 和 frame 的引用(Reference)，這樣函式內的修改會直接影響 main 裡面的變數
int InitInputAndDisplay(cv::VideoCapture& cap, cv::Mat& frame);

#endif // VIDEO_INIT_H
