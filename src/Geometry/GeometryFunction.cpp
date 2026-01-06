#include "GeometryFunction.h"
#include "CameraModel.h"
#include "GroundPlane.h"
#include "WorldProjector.h"
#include "TrackingBoxWorldTransformer.h" // 請確認你的資料夾路徑

#include <cstdio> // for sprintf or string formatting
#include <string>
#include <iomanip>
#include <sstream>

bool Draw_KPT_World = true;     //  true,   false
bool Draw_Box_World = true;

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

std::vector<TrackingBox> GeometryFunction(const cv::Mat& Src_frame, cv::Mat& Output_frame, const std::vector<TrackingBox>& TrackingResult, const CameraModel* cam)
{
    // 1. 複製原始影像到輸出影像
    if (Src_frame.empty()) {
        // 防止空圖傳入
        Output_frame = cv::Mat(); 
    } else {
        Output_frame = Src_frame.clone();
    }

    // // 2. 初始化相機與幾何模組
    // CameraModel cam;
    // // 建議：若效能敏感，應避免每次呼叫都讀檔，可改為傳入靜態 reference 或 singleton
    // if (!cam.loadFromYaml("../Camera-Config/Sensing-3M.yaml")) {
    //     fprintf(stderr, "Error: Failed to load camera config.\n");
    //     return {};
    // }

    GroundPlane plane = GroundPlane::Z0();
    WorldProjector projector(*cam, plane);
    TrackingBoxWorldTransformer tf(projector);

    std::vector<TrackingBox> WorldResult;
    WorldResult.reserve(TrackingResult.size());

    // 3. 處理每個物件
    for (const auto& tb : TrackingResult) {
        TrackingBox world_tb = tb;

        // --- A. 計算世界座標 ---
        // result_points 依 class_id 不同而含義不同：
        // class_id == 0: 對應 tb.kpts 的每個點
        // class_id > 0 : [0]為左下角, [1]為右下角
        std::vector<cv::Point3f> result_points = tf.toWorldPoints(tb);
        
        // 將計算結果存回輸出結構
        // world_tb.kpts = result_points;


        // centimeters to meters
        std::vector<cv::Point3f> converted_points;
        converted_points.reserve(result_points.size());

        for (const auto& p : result_points) {
            // 轉換公式
            float x_forward_m = p.y / 100.0f;      // 原 y(前,cm) -> 新 x(前,m)
            float y_left_m    = -(p.x / 100.0f);   // 原 x(右,cm) -> 新 y(左,m)
            // float z_height_m  = p.z / 100.0f;      // 假設 z 也是 cm，轉為 m (非必要，看用途)

            float confidence  = 1.0f;

            // 存入新點
            converted_points.emplace_back(x_forward_m, y_left_m, confidence);
        }

        // 將轉換後的點存回 world_tb
        world_tb.kpts = converted_points;
        WorldResult.push_back(world_tb);

        // --- B. 繪圖 ---
        if (!Output_frame.empty()) {
            
            if (tb.class_id == 0 && Draw_KPT_World == true) {
                // === 車道線 (Lane) ===
                // 遍歷所有關鍵點進行繪製
                for (size_t i = 0; i < tb.kpts.size(); ++i) {
                    if (i >= result_points.size()) break; // 安全檢查

                    // 取出原始像素座標 (tb.kpts 雖然是 Point3f，但在輸入時 x,y 是像素)
                    cv::Point2f px(tb.kpts[i].x, tb.kpts[i].y);
                    
                    // 繪製 (使用青色)
                    drawCoordinates(Output_frame, px, result_points[i], cv::Scalar(255, 255, 0));
                    
                    // 可選：畫個小圓點標記位置
                    cv::circle(Output_frame, px, 3, cv::Scalar(0, 255, 0), -1); 
                }
            } 
            if(tb.class_id != 0 && Draw_Box_World == true) {
                // === 物件 (Object) ===
                // 我們有計算 Box 左下與右下的世界座標，這裡選擇顯示左下角座標
                // 左下角像素位置
                cv::Point2f bl_px(static_cast<float>(tb.box.x), 
                                  static_cast<float>(tb.box.y + tb.box.height));
                
                // 右下角像素位置 (若你想顯示右下角)
                cv::Point2f br_px(static_cast<float>(tb.box.x + tb.box.width), 
                                  static_cast<float>(tb.box.y + tb.box.height));

                // 取得對應的世界座標 (Transformer 的實作順序：0是左下, 1是右下)
                if (result_points.size() >= 1) {
                    // 文字畫在 Box 下方一點點
                    cv::Point2f text_pos = bl_px + cv::Point2f(0, 15);
                    
                    // 繪製 (使用黃色)
                    drawCoordinates(Output_frame, text_pos, result_points[0], cv::Scalar(0, 255, 255));
                }
                
                // 繪製 Bounding Box 方便辨識
                cv::rectangle(Output_frame, tb.box, cv::Scalar(0, 0, 255), 2);
            }
        }
    }

    return WorldResult;
}

