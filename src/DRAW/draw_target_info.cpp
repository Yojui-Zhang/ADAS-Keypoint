#include <opencv2/opencv.hpp>
#include <iomanip> // 用於設定小數點位數
#include <sstream>

#include "draw_icon.h"

/**
 * @brief 在影像上方中間繪製 ADAS 資訊 (速度, 距離, TTC)
 * * @param img 輸入/輸出的影像 (cv::Mat)
 * @param speed 目標速度 (TargetSpeedKmh)
 * @param dist  目標距離 (Targetdistance)
 * @param ttc   碰撞時間 (TargetTTC)
 */
void DrawTargetInfo(cv::Mat& img, float speed, float dist, float ttc, int y) {
    if (img.empty()) return;

    // 1. 格式化字串 (使用 stringstream 控制小數點位數)
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1); // 設定小數點後 1 位
    ss << "Tg-Spd: " << speed << " km/h | Tg-Dist: " << dist << " m | Tg-TTC: " << ttc << " s";
    std::string text = ss.str();

    // 2. 設定字型參數
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;     // 字體大小
    int thickness = 2;          // 線條粗細
    int baseline = 0;

    // 3. 計算文字尺寸以便置中
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    // 4. 計算繪製座標 (X: 畫面寬度的一半減去文字寬度的一半, Y: 距離頂部 40 pixel)
    int x = (img.cols - textSize.width) / 2;
    // int y = 40; // 距離上方邊緣的像素距離，可依需求調整
    cv::Point textOrg(x, y);

    // 5. (選用) 繪製黑色背景框以增加對比度
    // 框的範圍比文字稍微大一點
    int padding = 5;
    cv::Rect bgRect(x - padding, y - textSize.height - padding, 
                    textSize.width + (padding * 2), textSize.height + baseline + (padding * 2));
    
    // 繪製半透明黑底 (這裡用實心黑底示範，若要半透明需用 addWeighted)
    cv::rectangle(img, bgRect, cv::Scalar(0, 0, 0), cv::FILLED);

    // 6. 決定文字顏色 (例如：若 TTC 小於 2.0 秒顯示紅色，否則綠色)
    cv::Scalar textColor = (ttc < 2.0f && ttc > 0.0f) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

    // 7. 繪製文字
    cv::putText(img, text, textOrg, fontFace, fontScale, textColor, thickness, cv::LINE_AA);
}