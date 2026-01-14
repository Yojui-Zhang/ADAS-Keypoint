// fileName: GeometryFunction.h
#pragma once
#include <vector>
#include <opencv2/opencv.hpp> // 為了 cv::Mat
#include "config.h" // 必須包含這個定義

class CameraModel;

// 輸入 TrackingResult (像素座標), 輸出 WorldResult (世界座標)
std::vector<TrackingBox> GeometryFunction(const cv::Mat& Src_frame, cv::Mat& Output_frame, std::vector<TrackingBox>& TrackingResult, const CameraModel* cam);