#include "input-view.h"
#include "config.h"
#ifdef _v4l2cap
    #include "V4L2_define.h"
    int v4l2res = v4l2init(V4L2_cap_num);
#endif

#ifdef _opengl
    extern unsigned char* outputRgbaMem;
#endif

#ifdef _v4l2cap
    extern cv::Mat v4l2Cam();
#endif


int InitInputAndDisplay(cv::VideoCapture& cap, cv::Mat& frame, const InputViewConfig& cfg) {
#ifdef _openCVcap

    if (cfg.camera_index >= 0) {
        cap.open(cfg.camera_index);
    } else {
        cap.open(cfg.video_path);
    }

    if (!cap.isOpened()) {
        printf("can't open openCV camera\n");
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
    if(outputRgbaMem == nullptr) {
        outputRgbaMem = (unsigned char*)calloc(cfg.capture_width * cfg.capture_height * 4, sizeof(unsigned char));
    }
    glinit();
#else
    cv::namedWindow(cfg.window_name, cv::WINDOW_NORMAL);
    if (cfg.fullscreen) {
        cv::setWindowProperty(cfg.window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }

#endif

    return 0;
}
