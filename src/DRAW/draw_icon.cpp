#include <opencv2/opencv.hpp>
#include <string>

#include "draw_icon.h"



std::unordered_map<Icon_ID, cv::Mat> IconManager::icons_;

namespace {

bool IsSpeedSignId(int sign_set)
{
    return sign_set >= 0 && sign_set <= 8;
}

Icon_ID SpeedSignIconId(int sign_set)
{
    switch (sign_set) {
        case 0: return Icon_ID::sign_100km;
        case 1: return Icon_ID::sign_110km;
        case 2: return Icon_ID::sign_30km;
        case 3: return Icon_ID::sign_40km;
        case 4: return Icon_ID::sign_50km;
        case 5: return Icon_ID::sign_60km;
        case 6: return Icon_ID::sign_70km;
        case 7: return Icon_ID::sign_80km;
        case 8: return Icon_ID::sign_90km;
        default: return Icon_ID::sign_30km;
    }
}

int ResolveDisplayedSpeedSignId(int sign_set)
{
    static int last_valid_sign_set = -1;
    if (IsSpeedSignId(sign_set)) {
        last_valid_sign_set = sign_set;
    }
    return last_valid_sign_set;
}

}  // namespace

cv::Mat Load_Icon(const std::string& path, cv::Size target){

    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "[IconManager] cannot load " << path << '\n';
        return {};
    }
    if (target.width > 0)
        cv::resize(img, img, target);
    return img;
}

cv::Size IconManager::getSize(Icon_ID id)
{
    auto it = icons_.find(id);
    return (it != icons_.end()) ? it->second.size() : cv::Size();
}

bool IconManager::Load_Picture(const std::string& dir)
{
    // light
    icons_[Icon_ID::Light_green]        = Load_Icon(dir + "/light/green_light.png",        {50,    50});
    icons_[Icon_ID::Light_yellow]       = Load_Icon(dir + "/light/yellow_light.png",       {50,    50});
    icons_[Icon_ID::Light_red]          = Load_Icon(dir + "/light/red_light.png",          {50,    50});
    icons_[Icon_ID::Lightgray]          = Load_Icon(dir + "/light/gray_light.png",         {50,    50});
    icons_[Icon_ID::Light_background]   = Load_Icon(dir + "/light/light_background.png",   {200,   112});

    // sign
    icons_[Icon_ID::sign_30km]  = Load_Icon(dir + "/sign/30km.png",     {100, 100});
    icons_[Icon_ID::sign_40km]  = Load_Icon(dir + "/sign/40km.png",     {100, 100});
    icons_[Icon_ID::sign_50km]  = Load_Icon(dir + "/sign/50km.png",     {100, 100});
    icons_[Icon_ID::sign_60km]  = Load_Icon(dir + "/sign/60km.png",     {100, 100});
    icons_[Icon_ID::sign_70km]  = Load_Icon(dir + "/sign/70km.png",     {100, 100});
    icons_[Icon_ID::sign_80km]  = Load_Icon(dir + "/sign/80km.png",     {100, 100});
    icons_[Icon_ID::sign_90km]  = Load_Icon(dir + "/sign/90km.png",     {100, 100});
    icons_[Icon_ID::sign_100km] = Load_Icon(dir + "/sign/100km.png",    {100, 100});
    icons_[Icon_ID::sign_110km] = Load_Icon(dir + "/sign/110km.png",    {100, 100});

    return true;
}

void IconManager::Draw_Icon(cv::Mat& bg, Icon_ID id, cv::Point loc)
{
    auto it = icons_.find(id);
    if (it == icons_.end() || it->second.empty()) return;

    const cv::Mat& icon = it->second;
    int ch = icon.channels();

    for (int y = 0; y < icon.rows && (loc.y + y) < bg.rows; ++y)
        for (int x = 0; x < icon.cols && (loc.x + x) < bg.cols; ++x)
        {
            if (ch == 4) {                          // BGRA with alpha
                cv::Vec4b p = icon.at<cv::Vec4b>(y,x);
                if (p[3])                           // alpha > 0
                    bg.at<cv::Vec3b>(loc.y+y, loc.x+x) = cv::Vec3b(p[0],p[1],p[2]);
            } else {
                bg.at<cv::Vec3b>(loc.y+y, loc.x+x) = icon.at<cv::Vec3b>(y,x);
            }
        }
}

cv::Mat IconManager::Draw_Icon_Light(cv::Mat& bgr, int light_Set){
/*
     light_set = 13      (Green Light)
     light_set = 16      (Yellow Light)
     light_set = 15      (Red Light)
     light_set = other  (Gray Light)
*/

    IconManager::Draw_Icon(bgr, Icon_ID::Light_background, cv::Point(1070,10));

    if (light_Set == 13){
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1090,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1146,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Light_green, cv::Point(1203,40));
    }
    else if(light_Set == 16){
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1090,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Light_yellow,cv::Point(1146,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1203,40));
    }
    else if(light_Set == 15){
        IconManager::Draw_Icon(bgr, Icon_ID::Light_red,   cv::Point(1090,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1146,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1203,40));
    }
    else{
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1090,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1146,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(1203,40));
    }

    return bgr;
}

cv::Mat IconManager::Draw_Icon_Sign(cv::Mat& bgr, int sign_Set){
    const int displayed_sign_set = ResolveDisplayedSpeedSignId(sign_Set);
    if (displayed_sign_set >= 0) {
        IconManager::Draw_Icon(bgr,
                               SpeedSignIconId(displayed_sign_set),
                               cv::Point(950,20));
    }

    return bgr;
}
