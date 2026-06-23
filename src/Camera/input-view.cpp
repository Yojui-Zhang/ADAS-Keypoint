#include "input-view.h"
#include "config.h"
#ifdef _v4l2cap
    #include "V4L2_define.h"
    int v4l2res = v4l2init(V4L2_cap_num);
#endif

#ifdef _opengl
    extern void glinit(void);
#endif

#ifdef _v4l2cap
    extern cv::Mat v4l2Cam();
#endif

namespace {

bool OpenCamera(cv::VideoCapture& cap, int camera_index) {
    if (camera_index < 0) {
        return false;
    }

    if (cap.open(camera_index, cv::CAP_V4L2)) {
        return true;
    }
    if (cap.open(camera_index)) {
        return true;
    }

    const std::string device_path = "/dev/video" + std::to_string(camera_index);
    if (cap.open(device_path, cv::CAP_V4L2)) {
        return true;
    }
    return cap.open(device_path);
}

}  // namespace


int InitInputAndDisplay(cv::VideoCapture& cap,
                        cv::Mat& frame,
                        const InputViewConfig& cfg,
                        bool use_opengl_display) {
#ifdef _openCVcap

    if (cfg.camera_index >= 0) {
        OpenCamera(cap, cfg.camera_index);
    } else {
        cap.open(cfg.video_path);
    }

    if (!cap.isOpened()) {
        if (cfg.camera_index >= 0) {
            std::cerr << "can't open openCV camera index " << cfg.camera_index
                      << " (tried CAP_V4L2 / default backend / /dev/videoN)" << std::endl;
        } else {
            std::cerr << "can't open input video: " << cfg.video_path << std::endl;
        }
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.capture_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.capture_height);

    cap >> frame;
    if(frame.empty()){
        printf("Failed to capture first frame\n");
        return -1;
    }

#endif

#ifdef _v4l2cap
    frame = v4l2Cam();
#endif

#ifdef _opengl
    if (use_opengl_display) {
        glinit();
    } else {
        cv::namedWindow(cfg.window_name, cv::WINDOW_NORMAL);
        if (cfg.fullscreen) {
            cv::setWindowProperty(cfg.window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        }
    }
#else
    cv::namedWindow(cfg.window_name, cv::WINDOW_NORMAL);
    if (cfg.fullscreen) {
        cv::setWindowProperty(cfg.window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }

#endif

    return 0;
}
