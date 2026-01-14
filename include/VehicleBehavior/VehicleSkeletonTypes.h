#pragma once
#include <opencv2/core.hpp>
#include <array>

namespace vehicle_skeleton {

// 你的 12 個點：上/中/下 三層，每層 4 點（LF, RF, LR, RR）
// layout 只是「索引映射」，可對應任意輸入順序
struct SkeletonKptLayout {
    // top layer
    int top_lf = -1, top_rf = -1, top_lr = -1, top_rr = -1;
    // mid layer
    int mid_lf = -1, mid_rf = -1, mid_lr = -1, mid_rr = -1;
    // bottom layer
    int bot_lf = -1, bot_rf = -1, bot_lr = -1, bot_rr = -1;

    // 若你的輸入 kpts 順序就是：
    static SkeletonKptLayout Default0123_4567_891011() {
        SkeletonKptLayout L;
        L.top_lf = 9; L.top_rf = 10; L.top_lr = 11; L.top_rr = 12;
        L.mid_lf = 4; L.mid_rf = 5; L.mid_lr = 6; L.mid_rr = 7;
        L.bot_lf = 0; L.bot_rf = 1; L.bot_lr = 2; L.bot_rr = 3;
        return L;
    }

    static SkeletonKptLayout Default3456_78910_12131415() {
        SkeletonKptLayout L;
        L.top_lf = 11; L.top_rf = 12; L.top_lr = 13; L.top_rr = 14;
        L.mid_lf = 6; L.mid_rf = 7; L.mid_lr = 8; L.mid_rr = 9;
        L.bot_lf = 2; L.bot_rf = 3; L.bot_lr = 4; L.bot_rr = 5;
        return L;
    }

    // 自訂映射：依序填入 12 個 index（同上順序語意）
    static SkeletonKptLayout FromIndexArray(const std::array<int,12>& idx) {
        SkeletonKptLayout L;
        L.top_lf = idx[0];  L.top_rf = idx[1];  L.top_lr = idx[2];  L.top_rr = idx[3];
        L.mid_lf = idx[4];  L.mid_rf = idx[5];  L.mid_lr = idx[6];  L.mid_rr = idx[7];
        L.bot_lf = idx[8];  L.bot_rf = idx[9];  L.bot_lr = idx[10]; L.bot_rr = idx[11];
        return L;
    }
};

struct SkeletonDrawParams {
    cv::Scalar color = cv::Scalar(0, 255, 0);
    int thickness = 2;
    int kpt_radius = 3;

    bool draw_kpts = true;
    bool draw_heading_arrow = true;
    bool draw_heading_text = true;
};

} // namespace vehicle_skeleton

