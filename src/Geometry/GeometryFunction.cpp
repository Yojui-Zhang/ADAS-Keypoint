#include "GeometryFunction.h"
#include "CameraModel.h"
#include "GroundPlane.h"
#include "WorldProjector.h"
#include "TrackingBoxWorldTransformer.h" // 請確認你的資料夾路徑

#include <cstdio> // for sprintf or string formatting
#include <string>
#include <iomanip>
#include <sstream>

bool Draw_KPT_World = false;     //  true,   false
bool Draw_Box_World = false;

static void drawCoordinates(cv::Mat& img, const cv::Point2f& px_pos, const cv::Point3f& w_pos, const cv::Scalar& color)
{
    // 檢查是否為 NaN (投影失敗的點不繪製)
    if (std::isnan(w_pos.x) || std::isnan(w_pos.y) || std::isnan(w_pos.z)) return;

    // 格式化字串：例如 "X: 12.5 Y: 3.2" (保留一位小數)
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << "(" << w_pos.x << "," << w_pos.y << ")";
    std::string text = oss.str();

    // 繪製文字 (字體大小與粗細可依需求調整)
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 1;
    
    // 繪製黑色邊框讓文字在任何背景都清楚
    cv::putText(img, text, px_pos, fontFace, fontScale, cv::Scalar(0,0,0), thickness + 2);
    // 繪製主要顏色文字
    cv::putText(img, text, px_pos, fontFace, fontScale, color, thickness);
}

std::vector<TrackingBox> GeometryFunction(const cv::Mat& Src_frame, cv::Mat& Output_frame, std::vector<TrackingBox>& TrackingResult, const CameraModel* cam)
{
    // 1. 複製原始影像到輸出影像
    if (Src_frame.empty()) {
        // 防止空圖傳入
        Output_frame = cv::Mat(); 
    } else {
        Output_frame = Src_frame.clone();
    }

    GroundPlane plane = GroundPlane::Z0();
    WorldProjector projector(*cam, plane);
    TrackingBoxWorldTransformer tf(projector);

    std::vector<TrackingBox> WorldResult;
    WorldResult.reserve(TrackingResult.size());

    // 2. 處理每個物件
    for (auto& tb : TrackingResult) {
        
        std::vector<cv::Point3f> result_points = tf.toWorldPoints(tb);
        TrackingBox world_tb = tb;

        // --- A. 計算世界座標 ---
        // kpts：轉換後寫回 world_tb.kpts
        world_tb.kpts.clear();
        world_tb.kpts.reserve(result_points.size());
        for (const auto& p : result_points) {
            float x_forward_m = p.y / 100.0f;
            float y_left_m    = -(p.x / 100.0f);
            float confidence  = 1.0f;
            world_tb.kpts.emplace_back(x_forward_m, y_left_m, confidence);
        }

        // World_box：也做同樣的座標/單位轉換後寫回 world_tb.World_box
        world_tb.World_box.clear();
        world_tb.World_box.reserve(tb.World_box.size());
        for (const auto& p : tb.World_box) {
            float x_forward_m = p.y / 100.0f;
            float y_left_m    = -(p.x / 100.0f);
            float confidence  = 1.0f;
            world_tb.World_box.emplace_back(x_forward_m, y_left_m, confidence);
        }

        WorldResult.push_back(world_tb);

    }

    return WorldResult;
}

