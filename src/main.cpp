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

#include "input-view.h"
#include "GeometryFunction.h"
#include "lane_keeping.h"
#include "AccApi.h"
#include "AccDebugDraw.h"
#include "draw_icon.h"


// CANBus
#include "canbus_recv.h"
#include "lib.h"
#include "terminal.h"
#include "pid_controller.h"

//Engine
#ifdef USE_TFLITE
#include "../Engine/TFlite/include/TFlite_main.h"
#endif

#ifdef USE_TENSORRT
#include "../Engine/TensorRT/include/TensorRT_main.hpp"
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

cv::Mat inputView(input_video_height, input_video_width, CV_8UC3);
cv::Mat frame(process_video_height, process_video_width, CV_8UC3);
cv::Mat Output_frame(process_video_height, process_video_width, CV_8UC3);

// ====================== CANBus Set ======================
/**
 * @SteerCtrlSwitch:	方向盤控制_開關
 * @targetAngle:		方向盤控制_輸入訊號。WM_SET_ANGLE: 指定角度數值 WM_SET_ANGULAR_VELO: 輸入角速度數值
*/
extern volatile int steerCtrlMode;
extern double targetAngle; // left 0.0 ~ -510.0 , right 0.0 ~ 510.0 
extern double deceleration; // 0.0 - 10.0

float target_speed = 0.f;

CAR CAN;
// ====================== CANBus Set ======================


int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <LanePose_Model_Path>  <Classify_Model_Path>" << std::endl;
    return 1;
  }

  const char* lanepose_model_path = argv[1];
  char* classify_model_path = argv[2];
  char* Icon_path = "../icon";

  std::vector<TrackingBox> TrackingResult;
  std::vector<TrackingBox> WorldResult;
  clock_t start, end;
  double system_time_used;

  Config config;

  CameraModel cam;
  if (!cam.loadFromYaml("../Camera-Config/Sensing-3M.yaml")) {
      std::cerr << "Main: Failed to load camera config." << std::endl;
      return -1;
  }
  // ============================== Input View ==============================
  cv::VideoCapture cap; // 宣告 cap 物件
  
  // 呼叫外部函式進行初始化，將 cap 和 frame 傳進去
  if (InitInputAndDisplay(cap, inputView) != 0) {
      std::cerr << "Video Initialization Failed." << std::endl;
      return -1;
  }

  cv::Rect roi_input_view(input_video_width - rect_video_width, input_video_height - rect_video_height, rect_video_width, rect_video_height);
  frame = inputView(roi_input_view);
  cv::resize(frame, frame, cv::Size(process_video_width, process_video_height));

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

#ifdef CANBUS__
  canbus_recv(CAN);

	canbus_ctrl_steer(1);     // Start/Stop ctrl Steer
  canbus_ctrl_dec(1);       // Start/Stop ctrl Brake

  cout << "target_speed = " << endl;
  cin >> target_speed;

  // org_target_speed = target_speed;

  cout << "init target_speed" << target_speed;

  pthread_t t_S3_v; // 宣告 pthread 變數
  pthread_t t_S3_dec; // 宣告 pthread 變數

  pthread_create(&t_S3_v, NULL, S3_speed_v, NULL);
  pthread_create(&t_S3_dec, NULL, S3_dec, NULL);

#endif

  while (1) {
    start = clock();

    // =============================== Camera =============================

#ifdef _openCVcap
    cap >> inputView;

    cv::Rect roi_input_view(input_video_width - rect_video_width, input_video_height - rect_video_height, rect_video_width, rect_video_height);
    frame = inputView(roi_input_view);
    cv::resize(frame, frame, cv::Size(process_video_width, process_video_height));

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

    WorldResult = GeometryFunction(Output_frame, Output_frame, TrackingResult, &cam);

    // ============================== Inference ===========================
    // Algorithm for LKA / ACC / AEB / Behavior Detection

#ifdef CANBUS__
    float ego_vehicle_speed = CAN.speed;
#else
    float ego_vehicle_speed = 30;
#endif

    // ------------------------ LKA ------------------------
    std::string dbg;
    targetAngle = lane_steering_step(WorldResult, ego_vehicle_speed, &dbg, Output_frame, Output_frame, &cam);

    // cout << "Steer: " << targetAngle << endl;
    // ------------------------ ACC ------------------------

    acc::ACC_SetEgoSpeedKmh(ego_vehicle_speed);
    auto cmd = acc::ACC_Run(WorldResult);

    acc::ACC_DrawTrackingBoxes(Output_frame, WorldResult, cmd);

    float speed_kmh  = cmd.speed_kmh;
    deceleration = cmd.brake_0_10;

    float TargetSpeedKmh = cmd.TargetSpeedKmh;
    float Targetdistance = cmd.Targetdistance;
    float TargetTTC      = cmd.TargetTTC;

    // cout << "\nbrake: " << deceleration << endl;
    // cout << "Speed: " << speed_kmh << endl << endl;

    // ------------------------ Behavior Detection ------------------------


    // ------------------------ Draw info ------------------------
    DrawTargetInfo(Output_frame, TargetSpeedKmh, Targetdistance, TargetTTC);    //目標車速 目標距離 目標TTC

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
    imageShow(output_video_width, output_video_height, outputRgbaMem);
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
