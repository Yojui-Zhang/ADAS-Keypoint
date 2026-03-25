#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "config.h"

// 我增加了 unit_a, unit_b, unit_c 的預設參數
// 如果你不傳入單位，它會預設使用原本的 km/h, m, s
void DrawTargetInfo(cv::Mat& img, 
                    float txt_a, float txt_b, float txt_c, 
                    int y, 
                    const std::string& label_a, 
                    const std::string& label_b, 
                    const std::string& label_c,
                    const std::string& unit_a = " km/h", 
                    const std::string& unit_b = " m", 
                    const std::string& unit_c = " s") {
    
    if (img.empty()) return;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(1);
    
    bool has_prev = false; // 用來判斷是否需要加分隔線 " | "

    // --- 變數 A ---
    if (!label_a.empty()) {
        ss << label_a << ": " << txt_a << unit_a;
        has_prev = true;
    }

    // --- 變數 B ---
    if (!label_b.empty()) {
        if (has_prev) ss << " | "; // 如果前面有內容，才加分隔線
        ss << label_b << ": " << txt_b << unit_b;
        has_prev = true;
    }

    // --- 變數 C ---
    if (!label_c.empty()) {
        if (has_prev) ss << " | ";
        ss << label_c << ": " << txt_c << unit_c;
    }
       
    std::string text = ss.str();

    // 如果所有標籤都是空的，就不繪製任何東西
    if (text.empty()) return;

    // --- 以下為繪圖設定 (保持不變) ---
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.55;
    int thickness = 1;
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    int x = (img.cols - textSize.width) / 2;
    cv::Point textOrg(x, y);

    int padding = 5;
    cv::Rect bgRect(x - padding, y - textSize.height - padding, 
                    textSize.width + (padding * 2), textSize.height + baseline + (padding * 2));
    
    // cv::rectangle(img, bgRect, cv::Scalar(0, 0, 0), cv::FILLED);

    // 顏色邏輯：
    // 如果使用了第三個參數 (TTC) 且數值小於 2.0，則顯示紅色；否則顯示綠色
    // (這裡加了 !label_c.empty() 判斷，避免只顯示變數A時卻變紅)
    cv::Scalar textColor = cv::Scalar(0, 255, 0);

    cv::putText(img, text, textOrg, fontFace, fontScale, BLACK, thickness+1, cv::LINE_AA);
    cv::putText(img, text, textOrg, fontFace, fontScale, WHITE, thickness, cv::LINE_AA);
    
}