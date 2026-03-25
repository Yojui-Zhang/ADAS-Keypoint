#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
#include "time_sync.h"
#include "runtime_log_manager.h"
#include "keypad.h"
#include "keypad_control.h"

#include "canbus_recv.h"
#include "lib.h"
#include "terminal.h"

#ifdef USE_TFLITE
#include "../Engine/TFlite/include/TFlite_main.h"
#endif

#ifdef USE_TENSORRT
#include "../Engine/TensorRT/include/TensorRT_main.hpp"
#endif

#ifdef _opengl
static unsigned char* outputRgbaMem;
extern void glinit(void);
extern void swap_egl(void);
extern void imageShow(int width, int height, unsigned char rgb[]);
#endif

#ifdef _v4l2cap
extern uint64_t v4l2_get_last_buffer_timestamp_ns();
#endif

using namespace std;
using namespace cv;

extern volatile int steerCtrlMode;
extern double targetAngle;
extern double deceleration;

float target_speed = 0.f;
float Targetdistance = 0.f;
CAR CAN;

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

bool LoadRuntimeConfigWithFallback(const std::string& requested_path,
                                   AdasSystemConfig& out_cfg,
                                   std::string& out_loaded_path,
                                   std::string& out_error) {
  std::vector<std::string> candidates;
  if (requested_path.empty() == false) {
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

void HandlePendingCommands(keypad::CommandSource& command_source,
                           keypad::RuntimeControlState& control_state) {
  while (true) {
    const user_command_mode_t cmd = command_source.Consume();
    if (cmd == CMD_NONE) {
      break;
    }

    std::string message;
    if (keypad::HandleCommand(cmd, &control_state, &message)) {
      keypad::SyncCanRuntimeState(control_state);
      if (message.empty() == false) {
        std::cout << "Main: " << message << std::endl;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs cli_args;
  if (ParseCliArgs(argc, argv, cli_args) == false) {
    std::cerr << "Usage: " << argv[0]
              << " <LanePose_Model_Path> <Classify_Model_Path> [System_Config_Path]" << std::endl;
    return 1;
  }

  AdasSystemConfig runtime_cfg;
  std::string cfg_path;
  std::string cfg_error;
  if (LoadRuntimeConfigWithFallback(cli_args.system_config_path, runtime_cfg, cfg_path, cfg_error) == false) {
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
    std::cout << "Main: real_car requested but CANBUS__ is disabled. Fallback to video mode." << std::endl;
    run_mode = RunMode::Video;
  }
#endif
  if (run_mode == RunMode::RealCar && runtime_cfg.input.camera_index < 0) {
    std::cout << "Main: real_car requires live camera. Override input.camera_index -> 0." << std::endl;
    runtime_cfg.input.camera_index = 0;
  }
  std::cout << "Main: Run mode -> " << RunModeName(run_mode) << std::endl;

  std::string time_sync_error;
  if (TimeSyncInit(&time_sync_error) == false) {
    std::cerr << "Main: time sync init failed: " << time_sync_error << std::endl;
    return -1;
  }

  std::cout << "Main: Time sync source -> " << TimeSyncClockSource()
            << (TimeSyncUsingPtp() ? " (PTP)" : " (fallback)") << std::endl;

  adas_log::RuntimeLogManager runtime_log_manager(runtime_cfg, cfg_path);
  std::string log_error;
  if (runtime_log_manager.Start(run_mode != RunMode::VirtualRoad, &log_error) == false) {
    std::cerr << "Main: " << log_error << std::endl;
    return -1;
  }
  if (runtime_log_manager.AblationRunning()) {
    std::cout << "Main: Ablation log -> " << runtime_log_manager.AblationOutputPath() << std::endl;
  } else {
    std::cout << "Main: Ablation logger disabled." << std::endl;
  }
  if (run_mode != RunMode::VirtualRoad) {
    if (runtime_log_manager.ResearchRunning()) {
      std::cout << "Main: Research log -> " << runtime_log_manager.ResearchOutputPath() << std::endl;
    } else {
      std::cout << "Main: Research logger disabled." << std::endl;
    }
  }

  if (run_mode == RunMode::VirtualRoad) {
    std::string sim_error;
    if (runtime_log_manager.RunVirtualRoadSimulation(&sim_error) == false) {
      std::cerr << "Main: virtual road simulation failed: " << sim_error << std::endl;
      return -1;
    }
    std::cout << "Main: virtual road simulation completed." << std::endl;
    runtime_log_manager.Stop();
    return 0;
  }

  CameraModel cam;
  if (cam.loadFromYaml(runtime_cfg.app.camera_yaml_path) == false) {
      std::cerr << "Main: Failed to load camera config: "
                << runtime_cfg.app.camera_yaml_path << std::endl;
      return -1;
  }

  cv::Mat input_view(runtime_cfg.input.capture_height, runtime_cfg.input.capture_width, CV_8UC3);
  cv::Mat frame(process_video_height, process_video_width, CV_8UC3);
  cv::Mat output_frame(process_video_height, process_video_width, CV_8UC3);

  cv::VideoCapture cap;
  if (InitInputAndDisplay(cap, input_view, runtime_cfg.input) != 0) {
      std::cerr << "Video Initialization Failed." << std::endl;
      return -1;
  }

  if (PrepareProcessFrame(input_view, frame) == false) {
      std::cerr << "Main: first frame preprocess failed." << std::endl;
      return -1;
  }

#ifdef Write_Video__
  write_video(output_video_width, output_video_height, output_video_fps,
              "Output_video.mp4");
#endif

#ifdef USE_TFLITE
  tflite_set_sort_config(runtime_cfg.sort, runtime_cfg.sort_keypoint);

  if (tflite_init(cli_args.lanepose_model_path, frame) == false) return -1;

  if (Classify_and_icon_init(cli_args.classify_model_path, runtime_cfg.app.icon_path.c_str()) == false) return -1;
#endif

#ifdef USE_TENSORRT
  trt_set_sort_config(runtime_cfg.sort, runtime_cfg.sort_keypoint);

  if (trt_init(cli_args.lanepose_model_path,
               cli_args.classify_model_path,
               runtime_cfg.app.icon_path.c_str(),
               trt_config) == false) {
    std::cerr << "TensorRT init failed" << std::endl;
    return -1;
  }
#endif

  const vehicle_skeleton::SkeletonKptLayout layout = ResolveSkeletonLayout(runtime_cfg);
  collision::CollisionAssist collision_assist(runtime_cfg.collision);

  keypad::CommandSource keypad_source;
  keypad::ReaderConfig keypad_reader_cfg;
  keypad_reader_cfg.enable_evdev = runtime_cfg.app.enable_keypad_evdev;
  keypad_reader_cfg.device_path = runtime_cfg.app.keypad_device_path;
  const bool keypad_evdev_ready = keypad_source.Start(keypad_reader_cfg);

#ifdef CANBUS__
  canbus_recv(CAN);
#endif

#ifdef CANBUS__
  const bool canbus_compiled = true;
#else
  const bool canbus_compiled = false;
#endif
  keypad::RuntimeControlState control_state =
      keypad::MakeInitialRuntimeControlState(runtime_cfg.app, canbus_compiled);
  keypad::SyncCanRuntimeState(control_state);

  clock_t start, end;
  uint64_t frame_index = 0;

  while (1) {
    start = clock();
    HandlePendingCommands(keypad_source, control_state);

    uint64_t frame_sync_ns = 0;
    uint64_t frame_hw_ns = 0;

#ifdef _openCVcap
    cap >> input_view;
    frame_sync_ns = TimeSyncNowNs();
    if (input_view.empty()) {
      std::cout << "Main: input stream ended." << std::endl;
      break;
    }

    if (PrepareProcessFrame(input_view, frame) == false) {
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

    std::vector<TrackingBox> tracking_result;
    std::vector<TrackingBox> world_result;

#ifdef USE_TFLITE
    tracking_result = tflite_run_frame(frame,
                                       output_frame,
                                       runtime_cfg.model.classify_model_width,
                                       runtime_cfg.model.classify_model_height,
                                       control_state.draw_inference_overlay);
#endif

#ifdef USE_TENSORRT
    tracking_result = trt_process_frame(frame,
                                        output_frame,
                                        trt_config,
                                        control_state.draw_inference_overlay);
#endif

    world_result = GeometryFunction(output_frame, output_frame, tracking_result, &cam);

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

    if (control_state.draw_acc_overlay) {
      acc::ACC_DrawTrackingBoxes(output_frame, world_result, cmd.acc_cmd);
    }

    const float target_speed_kmh = cmd.acc_cmd.TargetSpeedKmh;
    Targetdistance = cmd.acc_cmd.Targetdistance;
    const float target_ttc = cmd.acc_cmd.TargetTTC;

    targetAngle = cmd.steer_deg;
    target_speed = cmd.speed_kmh;
    deceleration = cmd.brake_0_10;

    std::vector<TrackingBox> world_before_behavior;
    if (runtime_log_manager.AblationRunning()) {
      world_before_behavior = world_result;
    }

    vehicle_skeleton::SkeletonDrawParams behavior_draw_params;
    behavior_draw_params.draw_kpts = control_state.draw_behavior_overlay;
    behavior_draw_params.draw_heading_arrow = control_state.draw_behavior_overlay;
    behavior_draw_params.draw_heading_text = control_state.draw_behavior_overlay;

    if (runtime_cfg.behavior.enable) {
      vehicle_skeleton::RunVehicleSkeletonAndHeading(output_frame,
                                                     output_frame,
                                                     world_result,
                                                     layout,
                                                     behavior_draw_params);
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

    if (control_state.draw_collision_overlay &&
        ca.warning && runtime_cfg.app.draw_collision_border) {
      cv::rectangle(output_frame,
                    cv::Point(0, 0),
                    cv::Point(output_frame.cols - 1, output_frame.rows - 1),
                    RED,
                    10);
    }

    if (control_state.draw_collision_overlay &&
        ca.warning && ca.threat_id >= 0 && runtime_cfg.app.draw_collision_target_box) {
      for (const auto& tb : world_result) {
        if (tb.id == ca.threat_id) {
          cv::rectangle(output_frame, tb.box, YELLOW, 10);
        }
      }
    }

    DrawTargetInfo(output_frame,
                   target_speed_kmh, Targetdistance, target_ttc, 40,
                   "Tg-sped", "Tg-dist", "TTC",
                   " km/h", " m", " s");

    DrawTargetInfo(output_frame,
                   target_speed, targetAngle, deceleration, 80,
                   "Our-Speed", "Angle", "Dec",
                   " km/h", " m", " s");

    DrawTargetInfo(output_frame,
                   0, 0, CAN.speed, 120,
                   "", "", " ",
                   "", "", " km/h");

    adas_log::FrameSnapshot log_snapshot;
    log_snapshot.frame_index = frame_index;
    log_snapshot.frame_sync_ns = frame_sync_ns;
    log_snapshot.frame_hw_ns = frame_hw_ns;
    log_snapshot.cmd_sync_ns = cmd_sync_ns;
    log_snapshot.dt_s = dt_s;
    log_snapshot.ego_speed_kmh = ego_vehicle_speed_kmh;
    log_snapshot.target_speed_kmh = target_speed;
    log_snapshot.target_distance_m = Targetdistance;
    log_snapshot.target_ttc_s = target_ttc;
    log_snapshot.world_before_behavior = runtime_log_manager.AblationRunning() ? &world_before_behavior : nullptr;
    log_snapshot.world_result = &world_result;
    log_snapshot.vehicle_cmd = &cmd;
    log_snapshot.collision_output = &ca;
#ifdef CANBUS__
    log_snapshot.can_valid = true;
#else
    log_snapshot.can_valid = false;
#endif
    log_snapshot.can_state = &CAN;
    runtime_log_manager.LogFrame(log_snapshot);
    frame_index += 1;

#ifdef Save_infer_raw_data__
    if (SaveOutputTensorToTxt(pose.interpreter.get(),
                              0,
                              "yolov8_output.txt") == false) {
      std::cerr << "SaveOutputTensorToTxt failed" << std::endl;
    }
#endif

    keypad::DrawRuntimeStatusOverlay(output_frame, control_state, keypad_evdev_ready);

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
    keypad_source.PushCvKey(key);
    HandlePendingCommands(keypad_source, control_state);

    if (key == 32) {
      std::cout << "Jump Out" << std::endl;
      break;
    }
  }

  keypad::ShutdownRuntimeControl(&control_state);
  keypad_source.Stop();

#ifdef _openCVcap
  cap.release();
#endif
#ifdef Write_Video__
  video_writer.release();
#endif
  runtime_log_manager.Stop();

  return 0;
}
