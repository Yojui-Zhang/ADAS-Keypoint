#include "runtime_log_manager.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>

#include "time_sync.h"

namespace {

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
  if (cfg_path.empty() == false && cfg.has_parent_path()) {
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

ablation::AlgorithmAblationOptions BuildAblationOptions(
    const AdasSystemConfig& runtime_cfg,
    const std::string& cfg_path) {
  ablation::AlgorithmAblationOptions options;
  options.output_path = runtime_cfg.ablation.output_path;
  options.output_dir = runtime_cfg.ablation.output_dir;
  options.flush_every_n = runtime_cfg.ablation.flush_every_n;
  options.plot_size_px = runtime_cfg.ablation.plot_size_px;
  options.plot_margin_px = runtime_cfg.ablation.plot_margin_px;
  options.steering_ratio = runtime_cfg.stability.steering_ratio;
  options.wheelbase_m = runtime_cfg.stability.wheelbase_m;
  options.virtual_road_enable = runtime_cfg.ablation.virtual_road_enable;
  options.virtual_road_mode = runtime_cfg.ablation.virtual_road_mode;
  options.virtual_road_csv_path =
      ResolvePathWithConfig(runtime_cfg.ablation.virtual_road_csv_path, cfg_path);
  options.virtual_road_length_m = runtime_cfg.ablation.virtual_road_length_m;
  options.virtual_road_step_m = runtime_cfg.ablation.virtual_road_step_m;
  options.virtual_road_lane_width_m = runtime_cfg.ablation.virtual_road_lane_width_m;
  options.virtual_road_arc_radius_m = runtime_cfg.ablation.virtual_road_arc_radius_m;
  options.virtual_road_s_amplitude_m = runtime_cfg.ablation.virtual_road_s_amplitude_m;
  options.virtual_road_s_wavelength_m = runtime_cfg.ablation.virtual_road_s_wavelength_m;
  options.enabled = runtime_cfg.ablation.enable;
  return options;
}

ResearchLogOptions BuildResearchOptions(const AdasSystemConfig& runtime_cfg) {
  ResearchLogOptions options;
  options.steering_ratio = runtime_cfg.stability.steering_ratio;
  options.wheelbase_m = runtime_cfg.stability.wheelbase_m;
  options.time_sync_uses_ptp = TimeSyncUsingPtp();
  options.time_sync_source = TimeSyncClockSource();
  return options;
}

void FillCanState(ResearchLogFrame* log_frame,
                  bool can_valid,
                  const CAR* can_state) {
  if (log_frame == nullptr) {
    return;
  }

  log_frame->can_valid = can_valid && can_state != nullptr;
  if (log_frame->can_valid == false) {
    return;
  }

  log_frame->can_speed_kmh = can_state->speed;
  log_frame->can_speed_raw_kmh = can_state->speedOri;
  log_frame->can_steer_deg = can_state->steer;
  log_frame->can_yaw_deg_s = can_state->yaw;
  log_frame->can_theta_deg = can_state->theta;
  log_frame->can_lat_accel_mps2 = can_state->latAccel;
  log_frame->can_long_accel_mps2 = can_state->longAccel;
  log_frame->can_steering_torque_nm = can_state->steeringTorque;
  log_frame->can_meterage_m = can_state->meterage;
  log_frame->can_throttle = can_state->throttle;
  log_frame->can_gear = can_state->gear;
  log_frame->can_turn_signal = can_state->turningSignal;
}

}  // namespace

namespace adas_log {

RuntimeLogManager::RuntimeLogManager(const AdasSystemConfig& runtime_cfg,
                                     const std::string& cfg_path)
    : runtime_cfg_(runtime_cfg),
      cfg_path_(cfg_path),
      ablation_logger_(BuildAblationOptions(runtime_cfg_, cfg_path_)),
      research_logger_(BuildResearchOptions(runtime_cfg_)) {}

RuntimeLogManager::~RuntimeLogManager() {
  Stop();
}

bool RuntimeLogManager::Start(bool enable_research_logger,
                              std::string* out_error) {
  std::string ablation_error;
  if (ablation_logger_.Start(&ablation_error) == false) {
    if (out_error != nullptr) {
      *out_error = "failed to start algorithm ablation logger: " + ablation_error;
    }
    return false;
  }

  if (enable_research_logger == false) {
    return true;
  }

  std::string research_error;
  if (research_logger_.Start(&research_error) == false) {
    ablation_logger_.Stop();
    if (out_error != nullptr) {
      *out_error = "failed to start research logger: " + research_error;
    }
    return false;
  }

  return true;
}

bool RuntimeLogManager::RunVirtualRoadSimulation(std::string* out_error) {
  ablation::VirtualRoadSimulationOptions sim_opts;
  sim_opts.frame_count = runtime_cfg_.ablation.virtual_sim_frame_count;
  sim_opts.dt_s = runtime_cfg_.ablation.virtual_sim_dt_s;
  sim_opts.speed_kmh = runtime_cfg_.ablation.virtual_sim_speed_kmh;
  sim_opts.max_steer_deg = runtime_cfg_.ablation.virtual_sim_max_steer_deg;
  sim_opts.vc_k_cte = runtime_cfg_.ablation.virtual_sim_vc_k_cte;
  sim_opts.vc_k_heading = runtime_cfg_.ablation.virtual_sim_vc_k_heading;
  sim_opts.raw_k_cte = runtime_cfg_.ablation.virtual_sim_raw_k_cte;
  sim_opts.raw_k_heading = runtime_cfg_.ablation.virtual_sim_raw_k_heading;
  sim_opts.raw_steer_bias_deg = runtime_cfg_.ablation.virtual_sim_raw_steer_bias_deg;
  sim_opts.raw_steer_osc_amp_deg = runtime_cfg_.ablation.virtual_sim_raw_steer_osc_amp_deg;
  sim_opts.raw_steer_osc_period_s = runtime_cfg_.ablation.virtual_sim_raw_steer_osc_period_s;
  return ablation_logger_.RunVirtualRoadSimulation(sim_opts, out_error);
}

void RuntimeLogManager::LogFrame(const FrameSnapshot& snapshot) {
  if (snapshot.vehicle_cmd == nullptr ||
      snapshot.world_result == nullptr ||
      snapshot.collision_output == nullptr) {
    return;
  }

  if (ablation_logger_.IsRunning()) {
    ablation::AlgorithmAblationFrame ablation_frame;
    ablation_frame.frame_index = snapshot.frame_index;
    ablation_frame.frame_sync_ns = snapshot.frame_sync_ns;
    ablation_frame.dt_s = snapshot.dt_s;
    ablation_frame.ego_speed_kmh = snapshot.ego_speed_kmh;
    ablation_frame.world_before_skeleton = snapshot.world_before_behavior;
    ablation_frame.world_after_skeleton = snapshot.world_result;
    ablation_frame.vehicle_control_cmd = *snapshot.vehicle_cmd;
    ablation_logger_.Step(ablation_frame);
  }

  if (research_logger_.IsRunning() == false) {
    return;
  }

  int world_car_count = 0;
  int world_person_count = 0;
  int world_rider_count = 0;
  for (const auto& tb : *snapshot.world_result) {
    if (tb.class_id == 1) world_car_count += 1;
    else if (tb.class_id == 2) world_rider_count += 1;
    else if (tb.class_id == 3) world_person_count += 1;
  }

  ResearchLogFrame log_frame;
  log_frame.frame_index = snapshot.frame_index;
  log_frame.frame_sync_ns = snapshot.frame_sync_ns;
  log_frame.frame_hw_ns = snapshot.frame_hw_ns;
  log_frame.cmd_sync_ns = snapshot.cmd_sync_ns;
  log_frame.can_steer_tx_sync_ns = TimeSyncGetCanSteerTxNs();
  log_frame.can_brake_tx_sync_ns = TimeSyncGetCanBrakeTxNs();

  log_frame.dt_s = snapshot.dt_s;
  log_frame.ego_speed_kmh = snapshot.ego_speed_kmh;

  log_frame.cmd_speed_kmh = snapshot.target_speed_kmh;
  log_frame.cmd_steer_deg = snapshot.vehicle_cmd->steer_deg;
  log_frame.cmd_brake_0_10 = snapshot.vehicle_cmd->brake_0_10;
  log_frame.lka_steer_deg_raw = snapshot.vehicle_cmd->lka_steer_deg_raw;

  log_frame.acc_target_speed_kmh = snapshot.vehicle_cmd->acc_cmd.TargetSpeedKmh;
  log_frame.acc_target_distance_m = snapshot.target_distance_m;
  log_frame.acc_target_ttc_s = snapshot.target_ttc_s;
  log_frame.acc_target_ttc_std_s = snapshot.vehicle_cmd->acc_cmd.TargetTTCStd;

  log_frame.collision_warning = snapshot.collision_output->warning;
  log_frame.collision_threat_id = snapshot.collision_output->threat_id;
  log_frame.collision_threat_ttc_s = snapshot.collision_output->threat_ttc_s;
  log_frame.collision_threat_dist_now_m = snapshot.collision_output->threat_dist_now_m;
  log_frame.collision_threat_min_dist_m = snapshot.collision_output->threat_min_dist_m;
  log_frame.collision_threat_approach_speed_mps = snapshot.collision_output->threat_approach_speed_mps;
  log_frame.collision_threat_pos_x_m = snapshot.collision_output->threat_pos.x;
  log_frame.collision_threat_pos_y_m = snapshot.collision_output->threat_pos.y;

  log_frame.world_object_count = static_cast<int>(snapshot.world_result->size());
  log_frame.world_car_count = world_car_count;
  log_frame.world_person_count = world_person_count;
  log_frame.world_rider_count = world_rider_count;

  FillCanState(&log_frame, snapshot.can_valid, snapshot.can_state);
  research_logger_.LogFrame(log_frame);
}

void RuntimeLogManager::Stop() {
  research_logger_.Stop();
  ablation_logger_.Stop();
}

}  // namespace adas_log
