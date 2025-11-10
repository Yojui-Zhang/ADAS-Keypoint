#include <opencv2/opencv.hpp>
#include <string>

enum class Icon_ID {
    Light_green, Light_yellow, Light_red, Lightgray, Light_background,
    sign_30km, sign_40km, sign_50km, sign_60km, sign_70km,
    sign_80km, sign_90km, sign_100km, sign_110km

};

class IconManager{
    public:
        /** 若要取圖標本身尺寸，可用 getSize(id) */
        static cv::Size getSize(Icon_ID id);

        /** 一次性載入所有 PNG（傳入圖標資料夾路徑）*/
        static bool Load_Picture(const std::string& icon_dir);

        /** 疊加 icon 至背景影像（支援透明 4 通道）*/
        static void Draw_Icon(cv::Mat& background, Icon_ID id, cv::Point location);

        static cv::Mat Draw_Icon_Light(cv::Mat& bgr, int light_Set);
        static cv::Mat Draw_Icon_Sign(cv::Mat& bgr, int light_Set);

    private:
        static std::unordered_map<Icon_ID, cv::Mat> icons_;
};

