#include <opencv2/opencv.hpp>
#include <string>

#include "draw_icon.h"



std::unordered_map<Icon_ID, cv::Mat> IconManager::icons_;

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
    icons_[Icon_ID::sign_30km]  = Load_Icon(dir + "/sign/30km.png",     {75, 75});
    icons_[Icon_ID::sign_40km]  = Load_Icon(dir + "/sign/40km.png",     {75, 75});
    icons_[Icon_ID::sign_50km]  = Load_Icon(dir + "/sign/50km.png",     {75, 75});
    icons_[Icon_ID::sign_60km]  = Load_Icon(dir + "/sign/60km.png",     {75, 75});
    icons_[Icon_ID::sign_70km]  = Load_Icon(dir + "/sign/70km.png",     {75, 75});
    icons_[Icon_ID::sign_80km]  = Load_Icon(dir + "/sign/80km.png",     {75, 75});
    icons_[Icon_ID::sign_90km]  = Load_Icon(dir + "/sign/90km.png",     {75, 75});
    icons_[Icon_ID::sign_100km] = Load_Icon(dir + "/sign/100km.png",    {75, 75});
    icons_[Icon_ID::sign_110km] = Load_Icon(dir + "/sign/110km.png",    {75, 75});

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

    IconManager::Draw_Icon(bgr, Icon_ID::Light_background, cv::Point(0,10));

    if (light_Set == 13){
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(20,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(76,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Light_green, cv::Point(133,40));
    }
    else if(light_Set == 16){
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(20,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Light_yellow,cv::Point(76,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(133,40));
    }
    else if(light_Set == 15){
        IconManager::Draw_Icon(bgr, Icon_ID::Light_red,   cv::Point(20,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(76,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(133,40));
    }
    else{
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(20,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(76,40));
        IconManager::Draw_Icon(bgr, Icon_ID::Lightgray,  cv::Point(133,40));
    }

    return bgr;
}

cv::Mat IconManager::Draw_Icon_Sign(cv::Mat& bgr, int sign_Set){
/*
     light_set = 0      (Green Light)
     light_set = 1      (Yellow Light)
     light_set = 2      (Red Light)
     light_set = other  (Gray Light)
*/

    if (sign_Set == 0){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_100km,  cv::Point(210,40));
    }
    else if(sign_Set == 1){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_110km,  cv::Point(210,40));
    }
    else if(sign_Set == 2){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_30km,  cv::Point(210,40));
    }
    else if(sign_Set == 3){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_40km,  cv::Point(210,40));
    }
    else if(sign_Set == 4){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_50km,  cv::Point(210,40));
    }
    else if(sign_Set == 5){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_60km,  cv::Point(210,40));
    }
    else if(sign_Set == 6){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_70km,  cv::Point(210,40));
    }
    else if(sign_Set == 7){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_80km,  cv::Point(210,40));
    }
    else if(sign_Set == 8){
        IconManager::Draw_Icon(bgr, Icon_ID::sign_90km,  cv::Point(210,40));
    }

    return bgr;
}