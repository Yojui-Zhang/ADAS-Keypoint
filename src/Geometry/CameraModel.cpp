#include "CameraModel.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <limits>

static cv::Mat toMat32F(const cv::Mat& m)
{
    cv::Mat out;
    m.convertTo(out, CV_32F);
    return out;
}

bool CameraModel::loadFromYaml(const std::string& yaml_path)
{
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    cv::Mat K, D, Rcw, tcw;
    fs["K"] >> K;
    fs["D"] >> D;
    fs["R_cw"] >> Rcw;
    fs["t_cw"] >> tcw;

    if (K.empty() || D.empty() || Rcw.empty() || tcw.empty()) return false;
    if (K.rows != 3 || K.cols != 3) return false;
    if (Rcw.rows != 3 || Rcw.cols != 3) return false;
    if (!((tcw.rows == 3 && tcw.cols == 1) || (tcw.rows == 1 && tcw.cols == 3))) return false;

    K_   = toMat32F(K);
    D_   = toMat32F(D);
    Rcw_ = toMat32F(Rcw);

    if (tcw.rows == 1 && tcw.cols == 3) tcw = tcw.t();
    tcw_ = toMat32F(tcw);

    ready_ = true;
    return true;
}

cv::Mat CameraModel::Rwc() const
{
    // Rwc = Rcw^T
    return Rcw_.t();
}

cv::Mat CameraModel::cameraCenterWorld() const
{
    // C_w = -Rwc * tcw
    cv::Mat Cw = -Rwc() * tcw_;
    return Cw; // 3x1
}

cv::Point2f CameraModel::undistortPixel(const cv::Point2f& px) const
{
    // Use undistortPoints with P=K to return pixel coordinates
    std::vector<cv::Point2f> src{px}, dst;
    cv::undistortPoints(src, dst, K_, D_, cv::noArray(), K_);
    return dst[0];
}

cv::Vec3f CameraModel::pixelToUnitRayCamera(const cv::Point2f& undist_px) const
{
    // dir_c = normalize(K^-1 [u v 1]^T)
    cv::Mat Kinv = K_.inv();
    cv::Mat uv1 = (cv::Mat_<float>(3,1) << undist_px.x, undist_px.y, 1.0f);
    cv::Mat dir = Kinv * uv1; // 3x1

    cv::Vec3f d(dir.at<float>(0), dir.at<float>(1), dir.at<float>(2));
    float n = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    if (n <= 1e-8f) return cv::Vec3f(0,0,1);
    return d * (1.0f / n);
}

cv::Point2f CameraModel::project3dToPixel(const cv::Point3f& pt_w) const
{
    if (!ready_) return cv::Point2f(-1, -1);

    std::vector<cv::Point3f> objectPoints = {pt_w};
    std::vector<cv::Point2f> imagePoints;

    // Rcw_ 是 3x3 旋轉矩陣，projectPoints 需要旋轉向量 (Rotation Vector)
    cv::Mat rvec;
    cv::Rodrigues(Rcw_, rvec);

    // 使用 OpenCV 內建投影 (包含 K 矩陣與 D 失真係數)
    cv::projectPoints(objectPoints, rvec, tcw_, K_, D_, imagePoints);

    if (imagePoints.empty()) return cv::Point2f(-1, -1);
    return imagePoints[0];
}