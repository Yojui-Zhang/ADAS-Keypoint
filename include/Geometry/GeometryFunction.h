// fileName: GeometryFunction.h
#pragma once
#include <vector>
#include <opencv2/opencv.hpp> // 為了 cv::Mat
#include "config.h" // 必須包含這個定義

class CameraModel;

struct GeometryConfig {
    bool draw_kpt_world = false;      // 是否繪製 keypoint 的世界座標資訊
    bool draw_box_world = false;      // 是否繪製 box 的世界座標資訊
    float world_unit_scale = 0.01f;   // 世界單位縮放（常用於 cm -> m，0.01 代表除以 100）
};

void Geometry_SetConfig(const GeometryConfig& cfg);
GeometryConfig Geometry_GetConfig();

// 只做座標轉換，不改動輸出影像；適合 log raw AI detection 或其他中間結果。
std::vector<TrackingBox> GeometryConvertTrackingResultToWorld(const std::vector<TrackingBox>& TrackingResult, const CameraModel* cam);

// 輸入 TrackingResult (像素座標), 輸出 WorldResult (世界座標)
std::vector<TrackingBox> GeometryFunction(const cv::Mat& Src_frame, cv::Mat& Output_frame, std::vector<TrackingBox>& TrackingResult, const CameraModel* cam);
