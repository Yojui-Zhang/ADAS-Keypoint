#pragma once
#include <opencv2/core.hpp>
#include <string>

class CameraModel
{
public:
    // Load:
    //   K (3x3), D (1xN), R_cw (3x3), t_cw (3x1)
    bool loadFromYaml(const std::string& yaml_path);

    const cv::Mat& K() const { return K_; }
    const cv::Mat& D() const { return D_; }
    const cv::Mat& Rcw() const { return Rcw_; }   // p_c = Rcw * p_w + tcw
    const cv::Mat& tcw() const { return tcw_; }
    int imageWidth() const { return image_width_; }
    int imageHeight() const { return image_height_; }
    cv::Size calibrationImageSize() const { return cv::Size(image_width_, image_height_); }

    // Derived:
    cv::Mat Rwc() const;              // = Rcw^T
    cv::Mat cameraCenterWorld() const; // C_w = -Rwc * tcw

    // Undistort a pixel (return in pixel coordinates, not normalized)
    cv::Point2f undistortPixel(const cv::Point2f& px) const;

    // Build a ray direction in camera frame from an undistorted pixel
    // dir_c is normalized (unit length)
    cv::Vec3f pixelToUnitRayCamera(const cv::Point2f& undist_px) const;

    cv::Point2f project3dToPixel(const cv::Point3f& pt_w) const;

private:
    cv::Mat K_;    // 3x3, CV_32F
    cv::Mat D_;    // 1xN, CV_32F
    cv::Mat Rcw_;  // 3x3, CV_32F
    cv::Mat tcw_;  // 3x1, CV_32F
    int image_width_ = 0;
    int image_height_ = 0;

    bool ready_ = false;
};
