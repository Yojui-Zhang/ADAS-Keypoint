// 基本函式
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// 自訂頭檔
#include "config.h"
#include "write_video.h"
#include "system_config.h"

#include "input-view.h"
#include "GeometryFunction.h"
#include "lane_keeping.h"
#include "AccApi.h"
#include "AccConfig.h"
#include "AccDebugDraw.h"
#include "VehicleControlApi.h"
#include "VehicleSkeletonAPI.h"
#include "draw_icon.h"
#include "CollisionAssistApi.h"
#include "research_data_logger.h"
#include "algorithm_ablation_logger.h"
#include "time_sync.h"

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

#ifdef _v4l2cap
extern uint64_t v4l2_get_last_buffer_timestamp_ns();
#endif

using namespace std;
using namespace cv;

// ======================================================================
// CANBus Set
// ======================================================================
/**
 * @SteerCtrlSwitch:\t方向盤控制_開關
 * @targetAngle:\t\t方向盤控制_輸入訊號。WM_SET_ANGLE: 指定角度數值 WM_SET_ANGULAR_VELO: 輸入角速度數值
*/
extern volatile int steerCtrlMode;
extern double targetAngle; // left 0.0 ~ -510.0 , right 0.0 ~ 510.0
extern double deceleration; // 0.0 - 10.0

float target_speed = 0.f;
float Targetdistance = 0.f;

CAR CAN;
acc::AccConfig ACCconfig;

namespace {

struct CliArgs {
  const char* lanepose_model_path = nullptr;
  char* classify_model_path = nullptr;
  std::string system_config_path;
};

bool ParseCliArgs(int argc, char** argv, CliArgs& out_args) {
  if (argc < 3) return false;

  out_args.lanepose_model_path = argv[1];
  out_args.classify_model_path = argv[2];

  if (argc >= 4) {
    out_args.system_config_path = argv[3];
  }
  return true;
}

enum class RunMode {
  Video,
  VirtualRoad,
  RealCar
};

std::string ToLowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

RunMode ParseRunMode(const std::string& s) {
  const std::string mode = ToLowerCopy(s);
  if (mode == "virtual_road" || mode == "virtual-road" || mode == "virtual") {
    return RunMode::VirtualRoad;
  }
  if (mode == "real_car" || mode == "real-car" || mode == "real") {
    return RunMode::RealCar;
  }
  return RunMode::Video;
}

const char* RunModeName(RunMode mode) {
  switch (mode) {
    case RunMode::VirtualRoad: return "virtual_road";
    case RunMode::RealCar: return "real_car";
    default: return "video";
  }
}

std::string ResolvePathWithConfig(const std::string& raw_path,
                                  const std::string& cfg_path) {
  if (raw_path.empty()) return raw_path;

  namespace fs = std::filesystem;
  const fs::path raw(raw_path);
  std::error_code ec;

  if (raw.is_absolute()) {
    return raw.lexically_normal().string();
  }

  if (fs::exists(raw, ec)) {
    return fs::absolute(raw, ec).lexically_normal().string();
  }

  const fs::path cfg(cfg_path);
  if (!cfg_path.empty() && cfg.has_parent_path()) {
    const fs::path cfg_dir = cfg.parent_path();
    const fs::path from_cfg_dir = cfg_dir / raw;
    if (fs::exists(from_cfg_dir, ec)) {
      return fs::absolute(from_cfg_dir, ec).lexically_normal().string();
    }

    const fs::path from_cfg_parent = cfg_dir / ".." / raw;
    if (fs::exists(from_cfg_parent, ec)) {
      return fs::absolute(from_cfg_parent, ec).lexically_normal().string();
    }
  }

  return raw_path;
}

bool LoadRuntimeConfigWithFallback(const std::string& requested_path,
                                   AdasSystemConfig& out_cfg,
                                   std::string& out_loaded_path,
                                   std::string& out_error) {
  std::vector<std::string> candidates;
  if (!requested_path.empty()) {
    candidates.push_back(requested_path);
  } else {
    candidates.push_back("../config/system_config.yaml");
    candidates.push_back("./config/system_config.yaml");
    candidates.push_back("config/system_config.yaml");
  }

  for (const auto& path : candidates) {
    std::string err;
    if (LoadSystemConfig(path, out_cfg, &err)) {
      out_loaded_path = path;
      return true;
    }
    out_error = err;
  }

  return false;
}

void ApplyTensorRtRuntimeConfig(const TensorRtRuntimeConfig& runtime_cfg, Config& trt_cfg) {
  trt_cfg.topk = runtime_cfg.topk;
  trt_cfg.score_thres = runtime_cfg.score_thres;
  trt_cfg.iou_thres = runtime_cfg.iou_thres;
  trt_cfg.num_labels = runtime_cfg.num_labels;
}

void ApplySubsystemConfig(const AdasSystemConfig& runtime_cfg) {
  Geometry_SetConfig(runtime_cfg.geometry);
  lane_keeping_set_control_config(runtime_cfg.lka);
  lane_keeping_reset_state();
  acc::ACC_SetConfig(runtime_cfg.acc);
  stability::VehicleControl_SetStabilityConfig(runtime_cfg.stability);
}

bool PrepareProcessFrame(const cv::Mat& input_view, cv::Mat& process_frame) {
  if (input_view.empty()) return false;

  const int roi_w = std::min(rect_video_width, input_view.cols);
  const int roi_h = std::min(rect_video_height, input_view.rows);
  const int roi_x = std::max(0, input_view.cols - roi_w);
  const int roi_y = std::max(0, input_view.rows - roi_h);

  const cv::Rect roi_input_view(roi_x, roi_y, roi_w, roi_h);
  cv::Mat roi = input_view(roi_input_view);
  cv::resize(roi, process_frame, cv::Size(process_video_width, process_video_height));
  return true;
}

vehicle_skeleton::SkeletonKptLayout ResolveSkeletonLayout(const AdasSystemConfig& runtime_cfg) {
  if (runtime_cfg.behavior.use_custom_layout) {
    return vehicle_skeleton::SkeletonKptLayout::FromIndexArray(runtime_cfg.behavior.custom_layout);
  }

#ifdef USE_TFLITE
  return vehicle_skeleton::SkeletonKptLayout::Default0123_4567_891011();
#else
  return vehicle_skeleton::SkeletonKptLayout::Default3456_78910_12131415();
#endif
}

}  // namespace

// ======================================================================
// Main Code
// ======================================================================
int main(int argc, char** argv) {
  CliArgs cli_args;
  if (!ParseCliArgs(argc, argv, cli_args)) {
    std::cerr << "Usage: " << argv[0]
              << " <LanePose_Model_Path> <Classify_Model_Path> [System_Config_Path]" << std::endl;
    return 1;
  }

  AdasSystemConfig runtime_cfg;
  std::string cfg_path;
  std::string cfg_error;
  if (!LoadRuntimeConfigWithFallback(cli_args.system_config_path, runtime_cfg, cfg_path, cfg_error)) {
    std::cerr << "Main: " << cfg_error << std::endl;
    return -1;
  }
  std::cout << "Main: Loaded config -> " << cfg_path << std::endl;

  ApplySubsystemConfig(runtime_cfg);

  Config trt_config;
  ApplyTensorRtRuntimeConfig(runtime_cfg.model.tensorrt, trt_config);

  RunMode run_mode = ParseRunMode(runtime_cfg.app.run_mode);
#ifndef CANBUS__
  if (run_mode == RunMode::RealCar) {
    std::cout << "Main: run_mode=real_car requested, but CANBUS__ is disabled. "
                 "Fallback to video mode."
              << std::endl;
    run_mode = RunMode::Video;
  }
#endif
  if (run_mode == RunMode::RealCar && runtime_cfg.input.camera_index < 0) {
    std::cout << "Main: run_mode=real_car requires live camera. "
                 "Override input.camera_index -> 0."
              << std::endl;
    runtime_cfg.input.camera_index = 0;
  }
  std::cout << "Main: Run mode -> " << RunModeName(run_mode) << std::endl;

  std::string time_sync_error;
  if (!TimeSyncInit(&time_sync_error)) {
    std::cerr << "Main: time sync init failed: " << time_sync_error << std::endl;
    return -1;
  }

  std::cout << "Main: Time sync source -> " << TimeSyncClockSource()
            << (TimeSyncUsingPtp() ? " (PTP)" : " (fallback)") << std::endl;

  ablation::AlgorithmAblationOptions ablation_options;
  ablation_options.output_path = runtime_cfg.ablation.output_path;
  ablation_options.output_dir = runtime_cfg.ablation.output_dir;
  ablation_options.flush_every_n = runtime_cfg.ablation.flush_every_n;
  ablation_options.plot_size_px = runtime_cfg.ablation.plot_size_px;
  ablation_options.plot_margin_px = runtime_cfg.ablation.plot_margin_px;
  ablation_options.steering_ratio = runtime_cfg.stability.steering_ratio;
  ablation_options.wheelbase_m = runtime_cfg.stability.wheelbase_m;
  ablation_options.virtual_road_enable = runtime_cfg.ablation.virtual_road_enable;
  ablation_options.virtual_road_mode = runtime_cfg.ablation.virtual_road_mode;
  ablation_options.virtual_road_csv_path =
      ResolvePathWithConfig(runtime_cfg.ablation.virtual_road_csv_path, cfg_path);
  ablation_options.virtual_road_length_m = runtime_cfg.ablation.virtual_road_length_m;
  ablation_options.virtual_road_step_m = runtime_cfg.ablation.virtual_road_step_m;
  ablation_options.virtual_road_lane_width_m = runtime_cfg.ablation.virtual_road_lane_width_m;
  ablation_options.virtual_road_arc_radius_m = runtime_cfg.ablation.virtual_road_arc_radius_m;
  ablation_options.virtual_road_s_amplitude_m = runtime_cfg.ablation.virtual_road_s_amplitude_m;
  ablation_options.virtual_road_s_wavelength_m = runtime_cfg.ablation.virtual_road_s_wavelength_m;
  ablation_options.enabled = runtime_cfg.ablation.enable;

  ablation::AlgorithmAblationLogger ablation_logger(ablation_options);
  std::string ablation_error;
  if (!ablation_logger.Start(&ablation_error)) {
    std::cerr << "Main: failed to start algorithm ablation logger: " << ablation_error << std::endl;
    return -1;
  }
  if (ablation_logger.IsRunning()) {
    std::cout << "Main: Ablation log -> " << ablation_logger.OutputPath() << std::endl;
  } else {
    std::cout << "Main: Ablation logger disabled." << std::endl;
  }

  if (run_mode == RunMode::VirtualRoad) {
    if (!ablation_logger.IsRunning()) {
      std::cerr << "Main: run_mode=virtual_road requires ablation.enable=1." << std::endl;
      return -1;
    }

    ablation::VirtualRoadSimulationOptions sim_opts;
    sim_opts.frame_count = runtime_cfg.ablation.virtual_sim_frame_count;
    sim_opts.dt_s = runtime_cfg.ablation.virtual_sim_dt_s;
    sim_opts.speed_kmh = runtime_cfg.ablation.virtual_sim_speed_kmh;
    sim_opts.max_steer_deg = runtime_cfg.ablation.virtual_sim_max_steer_deg;
    sim_opts.vc_k_cte = runtime_cfg.ablation.virtual_sim_vc_k_cte;
    sim_opts.vc_k_heading = runtime_cfg.ablation.virtual_sim_vc_k_heading;
    sim_opts.raw_k_cte = runtime_cfg.ablation.virtual_sim_raw_k_cte;
    sim_opts.raw_k_heading = runtime_cfg.ablation.virtual_sim_raw_k_heading;
    sim_opts.raw_steer_bias_deg = runtime_cfg.ablation.virtual_sim_raw_steer_bias_deg;
    sim_opts.raw_steer_osc_amp_deg = runtime_cfg.ablation.virtual_sim_raw_steer_osc_amp_deg;
    sim_opts.raw_steer_osc_period_s = runtime_cfg.ablation.virtual_sim_raw_steer_osc_period_s;

    std::string sim_error;
    if (!ablation_logger.RunVirtualRoadSimulation(sim_opts, &sim_error)) {
      std::cerr << "Main: virtual road simulation failed: " << sim_error << std::endl;
      return -1;
    }
    std::cout << "Main: virtual road simulation completed." << std::endl;
    ablation_logger.Stop();
    return 0;
  }

  CameraModel cam;
  if (!cam.loadFromYaml(runtime_cfg.app.camera_yaml_path)) {
      std::cerr << "Main: Failed to load camera config: "
                << runtime_cfg.app.camera_yaml_path << std::endl;
      return -1;
  }

  cv::Mat input_view(runtime_cfg.input.capture_height, runtime_cfg.input.capture_width, CV_8UC3);
  cv::Mat frame(process_video_height, process_video_width, CV_8UC3);
  cv::Mat output_frame(process_video_height, process_video_width, CV_8UC3);

  // ======================================================================
  // Input View Set
  // ======================================================================
  cv::VideoCapture cap;
  if (InitInputAndDisplay(cap, input_view, runtime_cfg.input) != 0) {
      std::cerr << "Video Initialization Failed." << std::endl;
      return -1;
  }

  if (!PrepareProcessFrame(input_view, frame)) {
      std::cerr << "Main: first frame preprocess failed." << std::endl;
      return -1;
  }

#ifdef Write_Video__
  write_video(output_video_width, output_video_height, output_video_fps,
              "Output_video.mp4");
#endif

  // ======================================================================
  // Engine Set
  // ======================================================================
#ifdef USE_TFLITE
  tflite_set_sort_config(runtime_cfg.sort, runtime_cfg.sort_keypoint);

  if (!tflite_init(cli_args.lanepose_model_path, frame)) return -1;

  if (!Classify_and_icon_init(cli_args.classify_model_path, runtime_cfg.app.icon_path.c_str())) return -1;
#endif

#ifdef USE_TENSORRT
  trt_set_sort_config(runtime_cfg.sort, runtime_cfg.sort_keypoint);

  if (!trt_init(cli_args.lanepose_model_path,
                cli_args.classify_model_path,
                runtime_cfg.app.icon_path.c_str(),
                trt_config)) {
    std::cerr << "TensorRT init failed\n";
    return -1;
  }
#endif

  const vehicle_skeleton::SkeletonKptLayout layout = ResolveSkeletonLayout(runtime_cfg);
  collision::CollisionAssist collision_assist(runtime_cfg.collision);

  ResearchLogOptions log_options;
  log_options.steering_ratio = runtime_cfg.stability.steering_ratio;
  log_options.wheelbase_m = runtime_cfg.stability.wheelbase_m;
  log_options.time_sync_uses_ptp = TimeSyncUsingPtp();
  log_options.time_sync_source = TimeSyncClockSource();

  ResearchDataLogger research_logger(log_options);
  std::string logger_error;
  if (!research_logger.Start(&logger_error)) {
    std::cerr << "Main: failed to start research logger: " << logger_error << std::endl;
    return -1;
  }
  if (research_logger.IsRunning()) {
    std::cout << "Main: Research log -> " << research_logger.OutputPath() << std::endl;
  } else {
    std::cout << "Main: Research logger disabled." << std::endl;
  }

  // ======================================================================
  // CANBus
  // ======================================================================

#ifdef CANBUS__
  canbus_recv(CAN);

  canbus_ctrl_steer(1);     // Start/Stop ctrl Steer
  canbus_ctrl_dec(1);       // Start/Stop ctrl Brake

  cout << "target_speed = " << endl;
  cin >> ACCconfig.cruise_speed_kmh;

  pthread_t t_S3_v; // 宣告 pthread 變數
  pthread_t t_S3_dec; // 宣告 pthread 變數

  pthread_create(&t_S3_v, NULL, S3_speed_v, NULL);
  pthread_create(&t_S3_dec, NULL, S3_dec, NULL);

#endif

  clock_t start, end;
  uint64_t frame_index = 0;

  while (1) {
    start = clock();

    // ======================================================================
    // Input View
    // ======================================================================
    uint64_t frame_sync_ns = 0;
    uint64_t frame_hw_ns = 0;

#ifdef _openCVcap
    cap >> input_view;
    frame_sync_ns = TimeSyncNowNs();
    if (input_view.empty()) {
      std::cout << "Main: input stream ended." << std::endl;
      break;
    }

    if (!PrepareProcessFrame(input_view, frame)) {
      std::cout << "Main: frame preprocess failed." << std::endl;
      break;
    }
#endif
#ifdef _v4l2cap
    frame = v4l2Cam();
    frame_hw_ns = v4l2_get_last_buffer_timestamp_ns();
    frame_sync_ns = TimeSyncNowNs();
    if (frame.empty()) {
      std::cout << "Main: v4l2 frame empty." << std::endl;
      break;
    }
#endif

    // ======================================================================
    // Inference
    // ======================================================================
    std::vector<TrackingBox> tracking_result;
    std::vector<TrackingBox> world_result;

#ifdef USE_TFLITE
    tracking_result = tflite_run_frame(frame,
                                       output_frame,
                                       runtime_cfg.model.classify_model_width,
                                       runtime_cfg.model.classify_model_height);
#endif

#ifdef USE_TENSORRT
    tracking_result = trt_process_frame(frame, output_frame, trt_config);
#endif

    world_result = GeometryFunction(output_frame, output_frame, tracking_result, &cam);

    // ======================================================================
    // Algorithm for LKA / ACC / Stability / Behavior / Collision
    // ======================================================================
#ifdef CANBUS__
    const float ego_vehicle_speed_kmh = CAN.speed;
#else
    const float ego_vehicle_speed_kmh = runtime_cfg.app.fallback_ego_speed_kmh;
#endif

    static auto last = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    float dt_s = std::chrono::duration<float>(now - last).count();
    last = now;
    dt_s = std::clamp(dt_s, 0.005f, 0.2f);

    const float ego_speed_mps = ego_vehicle_speed_kmh / 3.6f;

    std::string dbg;
    auto cmd = stability::VehicleControl_Run(world_result, ego_speed_mps, dt_s, &dbg);

    acc::ACC_DrawTrackingBoxes(output_frame, world_result, cmd.acc_cmd);

    const float target_speed_kmh = cmd.acc_cmd.TargetSpeedKmh;
    Targetdistance = cmd.acc_cmd.Targetdistance;
    const float target_ttc = cmd.acc_cmd.TargetTTC;

    targetAngle = cmd.steer_deg;
    target_speed = cmd.speed_kmh;
    deceleration = cmd.brake_0_10;

    std::vector<TrackingBox> world_before_behavior;
    if (ablation_logger.IsRunning()) {
      world_before_behavior = world_result;
    }

    if (runtime_cfg.behavior.enable) {
      vehicle_skeleton::RunVehicleSkeletonAndHeading(output_frame, output_frame, world_result, layout);
    }

    if (ablation_logger.IsRunning()) {
      ablation::AlgorithmAblationFrame ablation_frame;
      ablation_frame.frame_index = frame_index;
      ablation_frame.frame_sync_ns = frame_sync_ns;
      ablation_frame.dt_s = dt_s;
      ablation_frame.ego_speed_kmh = ego_vehicle_speed_kmh;
      ablation_frame.world_before_skeleton = &world_before_behavior;
      ablation_frame.world_after_skeleton = &world_result;
      ablation_frame.vehicle_control_cmd = cmd;
      ablation_logger.Step(ablation_frame);
    }

    auto ca = collision_assist.Step(
        world_result,
        ego_speed_mps,
        targetAngle,
        dt_s,
        runtime_cfg.app.enable_collision_actuation,
        &target_speed,
        &targetAngle,
        &deceleration);
    const uint64_t cmd_sync_ns = TimeSyncNowNs();

    if (ca.warning && runtime_cfg.app.draw_collision_border) {
      cv::rectangle(output_frame,
                    cv::Point(0, 0),
                    cv::Point(output_frame.cols - 1, output_frame.rows - 1),
                    RED,
                    10);
    }

    if (ca.warning && ca.threat_id >= 0 && runtime_cfg.app.draw_collision_target_box) {
      for (const auto& tb : world_result) {
        if (tb.id == ca.threat_id) {
          cv::rectangle(output_frame, tb.box, YELLOW, 10);
        }
      }
    }

    // ======================================================================
    // Draw info
    // ======================================================================
    DrawTargetInfo(output_frame,
                  target_speed_kmh, Targetdistance, target_ttc, 40,
                  "Tg-sped"     , "Tg-dist"     , "TTC",
                  " km/h"       , " m"          , " s");

    DrawTargetInfo(output_frame,
                  target_speed, targetAngle , deceleration, 80,
                  "Our-Speed" , "Angle"    , "Dec",
                  " km/h"     , " m"        , " s");

    DrawTargetInfo(output_frame,
                  0, 0 , CAN.speed, 120,
                  "" , "" , " ",
                  "" , "" , " km/h");

    int world_car_count = 0;
    int world_person_count = 0;
    int world_rider_count = 0;
    for (const auto& tb : world_result) {
      if (tb.class_id == 1) world_car_count += 1;
      else if (tb.class_id == 2) world_rider_count += 1;
      else if (tb.class_id == 3) world_person_count += 1;
    }

    ResearchLogFrame log_frame;
    log_frame.frame_index = frame_index++;
    log_frame.frame_sync_ns = frame_sync_ns;
    log_frame.frame_hw_ns = frame_hw_ns;
    log_frame.cmd_sync_ns = cmd_sync_ns;
    log_frame.can_steer_tx_sync_ns = TimeSyncGetCanSteerTxNs();
    log_frame.can_brake_tx_sync_ns = TimeSyncGetCanBrakeTxNs();

    log_frame.dt_s = dt_s;
    log_frame.ego_speed_kmh = ego_vehicle_speed_kmh;

    log_frame.cmd_speed_kmh = target_speed;
    log_frame.cmd_steer_deg = targetAngle;
    log_frame.cmd_brake_0_10 = deceleration;
    log_frame.lka_steer_deg_raw = cmd.lka_steer_deg_raw;

    log_frame.acc_target_speed_kmh = target_speed_kmh;
    log_frame.acc_target_distance_m = Targetdistance;
    log_frame.acc_target_ttc_s = target_ttc;
    log_frame.acc_target_ttc_std_s = cmd.acc_cmd.TargetTTCStd;

    log_frame.collision_warning = ca.warning;
    log_frame.collision_threat_id = ca.threat_id;
    log_frame.collision_threat_ttc_s = ca.threat_ttc_s;
    log_frame.collision_threat_dist_now_m = ca.threat_dist_now_m;
    log_frame.collision_threat_min_dist_m = ca.threat_min_dist_m;
    log_frame.collision_threat_approach_speed_mps = ca.threat_approach_speed_mps;
    log_frame.collision_threat_pos_x_m = ca.threat_pos.x;
    log_frame.collision_threat_pos_y_m = ca.threat_pos.y;

    log_frame.world_object_count = static_cast<int>(world_result.size());
    log_frame.world_car_count = world_car_count;
    log_frame.world_person_count = world_person_count;
    log_frame.world_rider_count = world_rider_count;

#ifdef CANBUS__
    log_frame.can_valid = true;
#else
    log_frame.can_valid = false;
#endif
    log_frame.can_speed_kmh = CAN.speed;
    log_frame.can_speed_raw_kmh = CAN.speedOri;
    log_frame.can_steer_deg = CAN.steer;
    log_frame.can_yaw_deg_s = CAN.yaw;
    log_frame.can_theta_deg = CAN.theta;
    log_frame.can_lat_accel_mps2 = CAN.latAccel;
    log_frame.can_long_accel_mps2 = CAN.longAccel;
    log_frame.can_steering_torque_nm = CAN.steeringTorque;
    log_frame.can_meterage_m = CAN.meterage;
    log_frame.can_throttle = CAN.throttle;
    log_frame.can_gear = CAN.gear;
    log_frame.can_turn_signal = CAN.turningSignal;

    research_logger.LogFrame(log_frame);

#ifdef Save_infer_raw_data__
    if (!SaveOutputTensorToTxt(pose.interpreter.get(), /*output_index=*/0,
                               "yolov8_output.txt")) {
      std::cerr << "SaveOutputTensorToTxt failed\n";
    }
#endif

#ifdef Write_Video__
    cv::resize(output_frame, output_frame,
               cv::Size(output_video_width, output_video_height));
    video_writer.write(output_frame);
#endif

    end = clock();
    if (runtime_cfg.app.show_timing_ms) {
      const double system_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
      cout << "Time taken: " << system_time_used << " ms" << endl;
    }

#ifdef _opengl
    outputRgbaMem = output_frame.data;
    imageShow(output_video_width, output_video_height, outputRgbaMem);
    swap_egl();
#else
    cv::resize(output_frame, output_frame,
               cv::Size(output_video_width, output_video_height));
    cv::imshow(runtime_cfg.input.window_name, output_frame);
#endif

    int key = cv::waitKey(runtime_cfg.app.wait_key_ms);
    if (key == 32) {
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
  ablation_logger.Stop();
  research_logger.Stop();

  return 0;
}
