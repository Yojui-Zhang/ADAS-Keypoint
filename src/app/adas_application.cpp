#include "adas_application.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cli_args.h"
#include "config.h"
#include "control_target_selector.h"
#include "frame_preprocessor.h"
#include "keypad_command_dispatch.h"
#include "lka_projection.h"
#include "run_mode.h"
#include "runtime_config.h"
#include "runtime_performance.h"
#include "skeleton_layout_resolver.h"
#include "system_config.h"
#include "write_video.h"
#include "adas_overlay.h"
#include "draw_commands.h"
#include "frame_presenter.h"

#include "input-view.h"
#include "GeometryFunction.h"
#include "CameraProjectionUtils.h"
#include "WorldGridOverlay.h"
#include "lane_keeping.h"
#include "lk_visualization.h"
#include "AccApi.h"
#include "AccConfig.h"
#include "AccDebugDraw.h"
#include "VehicleControlApi.h"
#include "VehicleSkeletonAPI.h"
#include "draw_icon.h"
#include "CollisionAssistApi.h"
#include "frame_snapshot_builder.h"
#include "runtime_log_bootstrap.h"
#include "time_sync.h"
#include "runtime_log_manager.h"
#include "runtime_control_overlay.h"
#include "runtime_control_runtime.h"
#include "runtime_control_state.h"
#include "keypad.h"
#include "sound/sound.h"

#include "canbus_recv.h"
#include "lib.h"
#include "terminal.h"

#ifdef USE_TFLITE
#include "TFlite_main.h"
#endif

#ifdef USE_TENSORRT
#include "TensorRT_main.hpp"
#endif

#ifdef _v4l2cap
extern uint64_t v4l2_get_last_buffer_timestamp_ns();
#endif

using namespace std;
using namespace cv;

extern volatile int steerCtrlMode;
extern volatile double targetAngle;
extern volatile double deceleration;
double deceleration_TEST;

float target_speed = 0.f;
float Targetdistance = 0.f;
CAR CAN;

namespace adas_app {

int RunAdasApplication(int argc, char** argv) {
  CliArgs cli_args;
  if (ParseCliArgs(argc, argv, cli_args) == false) {
    PrintCliUsage(std::cerr, argv[0]);
    std::cerr << std::endl;
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
    std::cout << "Main: real_car requires live camera. Override input.camera_index -> "
              << V4L2_cap_num << " (from config.h V4L2_cap_num)." << std::endl;
    runtime_cfg.input.camera_index = V4L2_cap_num;
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
  if (adas_log::StartAndReportRuntimeLogs(runtime_log_manager,
                                          run_mode != RunMode::VirtualRoad,
                                          run_mode == RunMode::VirtualRoad,
                                          std::cout,
                                          std::cerr) == false) {
    return -1;
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

  adas_render::FramePresenter frame_presenter(runtime_cfg.app.render_backend,
                                              runtime_cfg.input.window_name,
                                              output_video_width,
                                              output_video_height,
                                              runtime_cfg.app.wait_key_ms);
  const bool use_gpu_overlay_commands = frame_presenter.UsesOpenGl();

  cv::VideoCapture cap;
  if (InitInputAndDisplay(cap, input_view, runtime_cfg.input, frame_presenter.UsesOpenGl()) != 0) {
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
  controller::RuntimeControlState control_state =
      controller::MakeInitialRuntimeControlState(runtime_cfg.app, canbus_compiled);
  controller::SyncCanRuntimeState(control_state);

  WorldGridOverlayConfig world_grid_cfg;
  world_grid_cfg.enabled = runtime_cfg.app.draw_ground_grid_overlay;
  world_grid_cfg.forward_start_m = runtime_cfg.app.ground_grid_forward_start_m;
  world_grid_cfg.forward_end_m = runtime_cfg.app.ground_grid_forward_end_m;
  world_grid_cfg.lateral_min_m = runtime_cfg.app.ground_grid_lateral_min_m;
  world_grid_cfg.lateral_max_m = runtime_cfg.app.ground_grid_lateral_max_m;
  world_grid_cfg.spacing_m = runtime_cfg.app.ground_grid_spacing_m;
  world_grid_cfg.sample_step_m = runtime_cfg.app.ground_grid_sample_step_m;
  world_grid_cfg.major_every_n = runtime_cfg.app.ground_grid_major_every_n;
  world_grid_cfg.draw_labels = runtime_cfg.app.ground_grid_draw_labels;

  uint64_t frame_index = 0;

  while (1) {
    const auto perf_frame_start = PerfClock::now();
    HandlePendingCommands(keypad_source, control_state);
    const bool demo_presentation_mode = controller::DemoPresentationActive(control_state);
    const bool demo_lateral_control = controller::DemoLateralControlEnabled(control_state);
    const bool demo_longitudinal_control = controller::DemoLongitudinalControlEnabled(control_state);
    const bool demo_supervisor = controller::DemoSupervisorEnabled(control_state);
    const bool demo_lane_departure_warning =
        controller::DemoLaneDepartureWarningEnabled(control_state);
    const bool draw_required_demo_visuals = demo_presentation_mode;
    const bool draw_inference_overlay =
        control_state.draw_inference_overlay || draw_required_demo_visuals;

    RuntimePerformanceMetrics perf_metrics;
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

    const auto perf_after_input = PerfClock::now();
    perf_metrics.input_ms = ElapsedMs(perf_frame_start, perf_after_input);

    std::vector<TrackingBox> tracking_result;
    std::vector<TrackingBox> world_result;

    const auto inference_start = PerfClock::now();
#ifdef USE_TFLITE
    tracking_result = tflite_run_frame(frame,
                                       output_frame,
                                       runtime_cfg.model.classify_model_width,
                                       runtime_cfg.model.classify_model_height,
                                       draw_inference_overlay);
#endif

#ifdef USE_TENSORRT
    tracking_result = trt_process_frame(frame,
                                        output_frame,
                                        trt_config,
                                        draw_inference_overlay);
#endif

    const auto inference_end = PerfClock::now();
    perf_metrics.inference_ms = ElapsedMs(inference_start, inference_end);

    const auto geometry_start = PerfClock::now();
    world_result = GeometryFunction(output_frame, output_frame, tracking_result, &cam);
    const auto geometry_end = PerfClock::now();
    perf_metrics.geometry_ms = ElapsedMs(geometry_start, geometry_end);

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
    stability::VehicleControlOptions control_options;
    control_options.enable_lateral_control = demo_lateral_control;
    control_options.enable_longitudinal_control = demo_longitudinal_control;
    control_options.enable_supervisor = demo_supervisor;
    auto cmd = stability::VehicleControl_RunWithOptions(
        world_result, ego_speed_mps, dt_s, control_options, &dbg);
    perf_metrics.acc_scope_ms = cmd.perf.acc_scope_ms;
    perf_metrics.acc_ms = cmd.perf.acc_ms;
    perf_metrics.lka_ms = cmd.perf.lka_ms;
    perf_metrics.stability_ms = cmd.perf.stability_ms;
    perf_metrics.control_total_ms = cmd.perf.total_ms;

    const auto overlay_start = PerfClock::now();
    adas_render::DrawCommandBuffer overlay_commands;

    if (control_state.draw_acc_overlay) {
      acc::ACC_DrawTrackingBoxes(output_frame, world_result, cmd.acc_cmd);
      acc::ACC_DrawLongitudinalPhaseHud(output_frame, cmd.acc_cmd);
    }

    Targetdistance = cmd.acc_cmd.Targetdistance;
    const float target_ttc = cmd.acc_cmd.TargetTTC;

    double target_angle_cmd = cmd.steer_deg;
    target_speed = SelectActuatorSpeedTargetKmh(control_state, cmd, ego_vehicle_speed_kmh);
    double deceleration_cmd = cmd.brake_0_10;
    const float target_brake_0_10 = cmd.brake_0_10;

    const LkaReferenceSnapshot lka_reference_snapshot =
        lane_keeping_get_last_reference_snapshot();
    const cv::Point2f lka_ego_px =
        ProjectVehicleGroundPointToImage(cam, output_frame.size(), 0.0f, 0.0f);
    cv::Point2f lka_current_px;
    cv::Point2f lka_target_px;
    const bool lka_ego_px_valid =
        IsProjectedPointInsideImage(output_frame.size(), lka_ego_px, 0.0f);
    const bool lka_current_px_valid =
        ProjectVehicleGroundPointToPixel(cam, output_frame,
                                         lka_reference_snapshot.current_point,
                                         &lka_current_px);
    const bool lka_target_px_valid =
        ProjectVehicleGroundPointToPixel(cam, output_frame,
                                         lka_reference_snapshot.target_point,
                                         &lka_target_px);

    std::vector<TrackingBox> world_before_behavior;
    if (runtime_log_manager.AblationRunning()) {
      world_before_behavior = world_result;
    }

    vehicle_skeleton::SkeletonDrawParams behavior_draw_params;
    behavior_draw_params.draw_kpts = control_state.draw_behavior_overlay;
    behavior_draw_params.draw_heading_arrow = control_state.draw_behavior_overlay;
    behavior_draw_params.draw_heading_text = control_state.draw_behavior_overlay;

    const auto behavior_start = PerfClock::now();
    if (runtime_cfg.behavior.enable) {
      vehicle_skeleton::RunVehicleSkeletonAndHeading(output_frame,
                                                     output_frame,
                                                     world_result,
                                                     layout,
                                                     behavior_draw_params);
    }

    const auto behavior_end = PerfClock::now();
    perf_metrics.behavior_ms = ElapsedMs(behavior_start, behavior_end);

    const auto collision_start = PerfClock::now();
    auto ca = collision_assist.Step(
        world_result,
        ego_speed_mps,
        target_angle_cmd,
        dt_s,
        runtime_cfg.app.enable_collision_actuation && demo_supervisor,
        &target_speed,
        &target_angle_cmd,
        &deceleration_cmd);
    const auto collision_end = PerfClock::now();
    perf_metrics.collision_ms = ElapsedMs(collision_start, collision_end);
    const uint64_t cmd_sync_ns = TimeSyncNowNs();

    if (demo_lateral_control == false) {
      target_angle_cmd = 0.0;
    }
    if (demo_longitudinal_control == false) {
      target_speed = 0.0f;
      deceleration_cmd = 0.0;
    }

    if (control_state.draw_collision_overlay &&
        ca.warning && runtime_cfg.app.draw_collision_border) {
      const cv::Rect2f border_rect(0.0f,
                                   0.0f,
                                   static_cast<float>(output_frame.cols - 1),
                                   static_cast<float>(output_frame.rows - 1));
      if (use_gpu_overlay_commands) {
        overlay_commands.AddRectangle(border_rect, RED, 10.0f);
      } else {
        cv::rectangle(output_frame,
                      cv::Point(0, 0),
                      cv::Point(output_frame.cols - 1, output_frame.rows - 1),
                      RED,
                      10);
      }
    }

    if (control_state.draw_collision_overlay &&
        ca.warning && ca.threat_id >= 0 && runtime_cfg.app.draw_collision_target_box) {
      for (const auto& tb : world_result) {
        if (tb.id == ca.threat_id) {
          if (use_gpu_overlay_commands) {
            overlay_commands.AddRectangle(cv::Rect2f(static_cast<float>(tb.box.x),
                                                     static_cast<float>(tb.box.y),
                                                     static_cast<float>(tb.box.width),
                                                     static_cast<float>(tb.box.height)),
                                          YELLOW,
                                          10.0f);
          } else {
            cv::rectangle(output_frame, tb.box, YELLOW, 10);
          }
        }
      }
    }

    world_grid_cfg.enabled = control_state.draw_ground_grid_overlay;
    if (use_gpu_overlay_commands) {
      AppendWorldGridOverlayCommands(overlay_commands, output_frame.size(), cam, world_grid_cfg);
      if (world_grid_cfg.draw_labels) {
        DrawWorldGridOverlayLabels(output_frame, cam, world_grid_cfg);
      }
    } else {
      DrawWorldGridOverlay(output_frame, cam, world_grid_cfg);
    }

    target_angle_cmd = -target_angle_cmd;
    targetAngle = target_angle_cmd;
    deceleration = deceleration_cmd;
    // target_speed = cmd.speed_kmh;

    const float current_steer_deg =
#ifdef CANBUS__
        static_cast<float>(CAN.steer);
#else
        0.0f;
#endif
    const float target_brake_for_display =
        demo_longitudinal_control ? target_brake_0_10 : 0.0f;

    DrawTargetInfo(output_frame,
                   ego_vehicle_speed_kmh,
                   target_speed,
                   current_steer_deg,
                   static_cast<float>(target_angle_cmd),
                   static_cast<float>(deceleration_cmd),
                   target_brake_for_display);

    if (control_state.draw_lka_overlay || draw_required_demo_visuals) {
      lane_keeping::internal::DrawLkaLaneSolutionOnImage(
          world_result,
          output_frame,
          cam,
          lane_keeping_get_control_config(),
          0.0f,
          20.0f);
      if (use_gpu_overlay_commands) {
        adas_render::AppendLkaReferenceOverlayCommands(overlay_commands,
                                                       lka_ego_px,
                                                       lka_ego_px_valid,
                                                       lka_current_px,
                                                       lka_current_px_valid,
                                                       lka_target_px,
                                                       lka_target_px_valid);
        adas_render::DrawLkaReferenceOverlayLabels(output_frame,
                                                   lka_ego_px,
                                                   lka_ego_px_valid,
                                                   lka_current_px,
                                                   lka_current_px_valid,
                                                   lka_target_px,
                                                   lka_target_px_valid);
      } else {
        adas_render::DrawLkaReferenceOverlay(output_frame,
                                             lka_ego_px,
                                             lka_ego_px_valid,
                                             lka_current_px,
                                             lka_current_px_valid,
                                             lka_target_px,
                                             lka_target_px_valid);
      }
    }

    const bool draw_lane_detect_overlay =
        demo_presentation_mode ? demo_lane_departure_warning
                               : control_state.draw_lane_detect_overlay;
    if (draw_lane_detect_overlay) {
      const auto lane_departure_status =
          lane_keeping::internal::DetectRawLaneDepartureFromKeypoints(
              world_result,
              lane_keeping_get_control_config());
      if (lane_departure_status.departure) {
        sound::RequestLaneDepartureWarningSound();
      }
    }
    if (draw_lane_detect_overlay) {
      lane_keeping::internal::DrawLaneDetectOverlayOnImage(
          world_result,
          output_frame,
          cam,
          lane_keeping_get_control_config());
    }

#ifdef Save_infer_raw_data__
    if (SaveOutputTensorToTxt(pose.interpreter.get(),
                              0,
                              "yolov8_output.txt") == false) {
      std::cerr << "SaveOutputTensorToTxt failed" << std::endl;
    }
#endif

    controller::DrawRuntimeStatusOverlay(output_frame, control_state, keypad_evdev_ready);

    const auto overlay_end = PerfClock::now();
    perf_metrics.overlay_ms = ElapsedMs(overlay_start, overlay_end);
    perf_metrics.total_ms = ElapsedMs(perf_frame_start, overlay_end);
    perf_metrics.fps = perf_metrics.total_ms > 1e-6 ? (1000.0 / perf_metrics.total_ms) : 0.0;

    if (control_state.draw_status_hud) {
      adas_render::DrawPerformanceOverlay(output_frame, perf_metrics);
    }

    adas_log::FrameSnapshotBuilderInput log_input;
    log_input.frame_index = frame_index;
    log_input.frame_sync_ns = frame_sync_ns;
    log_input.frame_hw_ns = frame_hw_ns;
    log_input.cmd_sync_ns = cmd_sync_ns;
    log_input.dt_s = dt_s;
    log_input.ego_speed_kmh = ego_vehicle_speed_kmh;
    log_input.target_speed_kmh = target_speed;
    log_input.target_distance_m = Targetdistance;
    log_input.target_ttc_s = target_ttc;
    log_input.lka_reference_snapshot = &lka_reference_snapshot;
    log_input.lka_current_image_valid = lka_current_px_valid;
    log_input.lka_current_px = lka_current_px;
    log_input.lka_target_image_valid = lka_target_px_valid;
    log_input.lka_target_px = lka_target_px;
    log_input.tracking_result = &tracking_result;
    log_input.world_before_behavior = runtime_log_manager.AblationRunning() ? &world_before_behavior : nullptr;
    log_input.world_result = &world_result;
    log_input.vehicle_cmd = &cmd;
    log_input.collision_output = &ca;
#ifdef CANBUS__
    log_input.can_valid = true;
#else
    log_input.can_valid = false;
#endif
    log_input.can_state = &CAN;
    log_input.perf = &perf_metrics;

    const adas_log::FrameSnapshot log_snapshot = adas_log::BuildFrameSnapshot(log_input);
    runtime_log_manager.LogFrame(log_snapshot);
    frame_index += 1;

#ifdef Write_Video__
    cv::Mat video_frame =
        (use_gpu_overlay_commands && overlay_commands.Empty() == false) ? output_frame.clone() : output_frame;
    if (use_gpu_overlay_commands) {
      adas_render::DrawCommandsOpenCv(video_frame, overlay_commands);
    }
    cv::resize(video_frame, video_frame,
               cv::Size(output_video_width, output_video_height));
    video_writer.write(video_frame);
#endif

    if (runtime_cfg.app.show_timing_ms) {
      cout << "Time taken: " << perf_metrics.total_ms
           << " ms, FPS: " << perf_metrics.fps << endl;
    }

    int key = frame_presenter.Show(
        output_frame,
        (use_gpu_overlay_commands && overlay_commands.Empty() == false) ? &overlay_commands : nullptr);
    keypad_source.PushCvKey(key);
    HandlePendingCommands(keypad_source, control_state);

    if (key == 32) {
      std::cout << "Jump Out" << std::endl;
      break;
    }
  }

  controller::ShutdownRuntimeControl(&control_state);
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

}  // namespace adas_app
