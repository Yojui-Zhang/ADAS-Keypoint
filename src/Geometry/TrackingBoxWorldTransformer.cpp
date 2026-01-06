#include "TrackingBoxWorldTransformer.h"
std::vector<cv::Point3f> TrackingBoxWorldTransformer::toWorldPoints(const TrackingBox& tb) const
{
    std::vector<cv::Point3f> out;

    if (tb.class_id == 0) {
        // === Lane Line 處理 ===
        // 假設輸入的 kpts 雖然是 Point3f，但實際上存的是 (u, v, score/ignored)
        out.reserve(tb.kpts.size());
        for (const auto& kp : tb.kpts) {
            cv::Point2f px(kp.x, kp.y); // 取出像素座標
            cv::Point3f Xw;
            
            // 投影到地面
            bool ok = projector_.pixelToWorldOnPlane(px, Xw);
            
            // 即使失敗也塞入 NaN 保持點數對應 (方便後續處理)
            out.push_back(ok ? Xw : WorldProjector::NaNPoint());
        }
        return out;
    }

    // === Object Detection 處理 ===
    // 計算 Bounding Box 底邊的左下與右下角點
    // box = [x, y, w, h]
    const float x = static_cast<float>(tb.box.x);
    const float y = static_cast<float>(tb.box.y);
    const float w = static_cast<float>(tb.box.width);
    const float h = static_cast<float>(tb.box.height);

    // 定義底邊兩點 (通常用於距離估測)
    cv::Point2f bl(x,     y + h);
    cv::Point2f br(x + w, y + h);

    out.reserve(2);

    cv::Point3f Xw_bl, Xw_br;
    bool ok1 = projector_.pixelToWorldOnPlane(bl, Xw_bl);
    bool ok2 = projector_.pixelToWorldOnPlane(br, Xw_br);

    out.push_back(ok1 ? Xw_bl : WorldProjector::NaNPoint());
    out.push_back(ok2 ? Xw_br : WorldProjector::NaNPoint());
    
    return out;
}