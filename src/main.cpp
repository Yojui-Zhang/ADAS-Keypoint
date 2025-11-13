// 基本函式
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// 時間函式
#include <ctime>

// 信號處理（如 ctrl+c 可選）
#include <csignal>

// 自訂頭檔
#include "config.h"
#include "write_video.h"

#ifdef USE_TFLITE
#include "../Engine/TFlite/include/TFlite_main.h"
#endif

#ifdef USE_TENSORRT
#include "../Engine/TensorRT/include/TensorRT_main.hpp"
#endif

#ifdef _v4l2cap
#include "V4L2_define.h"
int v4l2res = v4l2init(V4L2_cap_num);
#endif

#ifdef _opengl
static unsigned char* outputRgbaMem;
extern void glinit(void);    // 初始化OpenGL
extern void swap_egl(void);  // 使用EGL顯示畫面
extern void imageShow(int width, int height,
                      unsigned char rgb[]);  // OpenGL打畫面
#endif

using namespace std;
using namespace cv;

cv::Mat frame(input_video_height, input_video_width, CV_8UC3);
cv::Mat Output_frame(input_video_height, input_video_width, CV_8UC3);

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <LanePose_Model_Path>  <Classify_Model_Path>" << std::endl;
    return 1;
  }

  const char* lanepose_model_path = argv[1];
  char* classify_model_path = argv[2];
  char* Icon_path = "../icon";

  Config config;

// ============================== Input View ==============================
#ifdef _openCVcap
  // 初始化影片路徑
  const char* inputVideoPath = "../video/1280x720/vecow-demo.mp4";

  // 打開輸入影片
  cv::VideoCapture cap(inputVideoPath);
  // cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    printf("can't open openCV camera\n");
    return -1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH,
          input_video_width);  // Setting the width of the video
  cap.set(cv::CAP_PROP_FRAME_HEIGHT,
          input_video_height);  // Setting the height of the video//

  int codec = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

  cap >> frame;
  cv::resize(frame, frame, cv::Size(input_video_width, input_video_height));

#endif
#ifdef _v4l2cap
  frame = v4l2Cam();
#endif
#ifdef _opengl
  outputRgbaMem = (unsigned char*)calloc(1280 * 720 * 4, sizeof(unsigned char));
  glinit();
#else
  cv::namedWindow("Screen", cv::WINDOW_NORMAL);
  cv::setWindowProperty("Screen", cv::WND_PROP_FULLSCREEN,
                        cv::WINDOW_FULLSCREEN);
#endif
  // ======================================================================

#ifdef Write_Video__
  write_video(output_video_width, output_video_height, output_video_fps,
              "Output_video.mp4");
#endif

// ============================== Engine Set ==============================

#ifdef USE_TFLITE
  if (!tflite_init(lanepose_model_path, frame)) return -1;

  if (!Classify_and_icon_init(classify_model_path, Icon_path)) return -1;
#endif

#ifdef USE_TENSORRT
  if (!trt_init(lanepose_model_path, classify_model_path, Icon_path, config)) {
    std::cerr << "TensorRT init failed\n";
    return -1;
  }
#endif

// ========================================================================
  std::vector<TrackingBox> TrackingResult;
  clock_t start, end;
  double system_time_used;

  while (1) {
    start = clock();

    // =============================== Camera =============================

#ifdef _openCVcap
    cap >> frame;
#endif
#ifdef _v4l2cap
    frame = v4l2Cam();
#endif

    // ============================== Inference ===========================
    // SORT Tracking / Draw icon / Draw object

#ifdef USE_TFLITE

    int classify_model_width = Classify_Model_Width;
    int classify_model_height = Classify_Model_Height;

    TrackingResult = tflite_run_frame(frame, Output_frame, classify_model_width,
                                        classify_model_height);

#endif

#ifdef USE_TENSORRT

    TrackingResult = trt_process_frame(frame, Output_frame, config);
#endif

    // ============================== Inference ===========================
    // Algorithm for LKA / ACC / AEB / Behavior Detection

    // ============================= Experiment =============================

#ifdef Save_infer_raw_data__
    if (!SaveOutputTensorToTxt(pose.interpreter.get(), /*output_index=*/0,
                               "yolov8_output.txt")) {
      std::cerr << "SaveOutputTensorToTxt failed\n";
    }
#endif

#ifdef Write_Video__
    cv::resize(Output_frame, Output_frame,
               cv::Size(output_video_width, output_video_height));
    video_writer.write(Output_frame);
#endif

    // ==============================================================

    end = clock();
    system_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    cout << "Time taken: " << system_time_used << " ms" << endl;

    // ==============================================================

#ifdef _opengl
    outputRgbaMem = Output_frame.data;
    imageShow(1280, 720, outputRgbaMem);
    swap_egl();
#else
    cv::resize(Output_frame, Output_frame,
               cv::Size(output_video_width, output_video_height));
    cv::imshow("Screen", Output_frame);
#endif

    int key = cv::waitKey(30);  // 等待 30 毫秒
    if (key == 32) {            // 空格鍵的 ASCII 代碼為 32
      std::cout << "Jump Out !" << std::endl;
      break;
    }
  }

  // 關閉資源
#ifdef _openCVcap
  cap.release();
#endif
#ifdef Write_Video__
  video_writer.release();
#endif

  return 0;
}
