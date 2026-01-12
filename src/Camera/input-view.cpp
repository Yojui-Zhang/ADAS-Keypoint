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


int InitInputAndDisplay(cv::VideoCapture& cap, cv::Mat& frame) {
#ifdef _openCVcap

    const char* inputVideoPath = "../video/1280x720/vecow-demo.mp4";

    cap.open(inputVideoPath);
    // cap.open(8);

    if (!cap.isOpened()) {
        printf("can't open openCV camera\n");
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, input_video_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, input_video_height);

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
        outputRgbaMem = (unsigned char*)calloc(1280 * 720 * 4, sizeof(unsigned char));
    }
    glinit();
#else
    cv::namedWindow("Screen", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Screen", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

#endif

    return 0;
}