#include "runtime_log_manager.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
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
  options.run_mode = runtime_cfg.app.run_mode;
#ifdef _v4l2cap
  options.input_source = "v4l2cap";
#else
  options.input_source = "openCVcap";
#endif
  options.can_tx_master_enable = runtime_cfg.app.can_tx_master_enable;
  options.can_throttle_enable =
      runtime_cfg.app.can_throttle_enable || runtime_cfg.app.can_longitudinal_enable;
  options.can_brake_enable =
      runtime_cfg.app.can_brake_enable || runtime_cfg.app.can_longitudinal_enable;
  options.can_longitudinal_enable =
      options.can_throttle_enable || options.can_brake_enable;
  options.can_steering_enable = runtime_cfg.app.can_steering_enable;
  return options;
}

std::string FormatLogFloat(double value) {
  if (!std::isfinite(value)) return "nan";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << value;
  return oss.str();
}

double TimeDeltaMsOrNaN(uint64_t newer_ns, uint64_t older_ns) {
  if (newer_ns == 0 || older_ns == 0 || newer_ns < older_ns) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(newer_ns - older_ns) * 1e-6;
}

const char* ClassIdName(int class_id) {
  switch (class_id) {
    case 0: return "roadlane";
    case 1: return "car";
    case 2: return "rider";
    case 3: return "person";
    case 4: return "light";
    case 5: return "signC";
    case 6: return "signT";
    default: return "other";
  }
}

struct ObjectLogSummary {
  int total_count = 0;
  int lane_count = 0;
  int car_count = 0;
  int rider_count = 0;
  int person_count = 0;
  int light_count = 0;
  int signc_count = 0;
  int signt_count = 0;
  int other_count = 0;
  std::string object_summary;
};

void IncrementObjectCount(ObjectLogSummary* summary, int class_id) {
  if (summary == nullptr) {
    return;
  }

  summary->total_count += 1;
  switch (class_id) {
    case 0: summary->lane_count += 1; break;
    case 1: summary->car_count += 1; break;
    case 2: summary->rider_count += 1; break;
    case 3: summary->person_count += 1; break;
    case 4: summary->light_count += 1; break;
    case 5: summary->signc_count += 1; break;
    case 6: summary->signt_count += 1; break;
    default: summary->other_count += 1; break;
  }
}

std::string BuildObjectStateSummary(const std::vector<TrackingBox>& objects,
                                    bool include_world_details) {
  std::ostringstream oss;
  bool first = true;

  for (const auto& tb : objects) {
    const double bottom_center_u = tb.box.x + 0.5 * tb.box.width;
    const double bottom_center_v = tb.box.y + tb.box.height;

    if (!first) {
      oss << ';';
    }
    first = false;

    oss << "id=" << tb.id
        << "|frame=" << tb.frame
        << "|cls=" << tb.class_id
        << "|label=" << ClassIdName(tb.class_id)
        << "|score=" << FormatLogFloat(tb.score)
        << "|classify=" << tb.classify_num
        << "|box_x=" << tb.box.x
        << "|box_y=" << tb.box.y
        << "|box_w=" << tb.box.width
        << "|box_h=" << tb.box.height
        << "|u_px=" << FormatLogFloat(bottom_center_u)
        << "|v_px=" << FormatLogFloat(bottom_center_v)
        << "|kpt_n=" << tb.kpts.size()
        << "|hist_n=" << tb.track_box_history.size();

    if (!include_world_details) {
      continue;
    }

    cv::Point2f ground_xy;
    const bool ground_valid = acc::TryGetGroundBottomCenterXY(tb, ground_xy);
    oss << "|x_m="
        << FormatLogFloat(ground_valid ? ground_xy.x : std::numeric_limits<double>::quiet_NaN())
        << "|y_m="
        << FormatLogFloat(ground_valid ? ground_xy.y : std::numeric_limits<double>::quiet_NaN());

    if (tb.World_box.size() >= 2) {
      const auto& bl = tb.World_box[0];
      const auto& br = tb.World_box[1];
      oss << "|bl_x_m=" << FormatLogFloat(bl.x)
          << "|bl_y_m=" << FormatLogFloat(bl.y)
          << "|br_x_m=" << FormatLogFloat(br.x)
          << "|br_y_m=" << FormatLogFloat(br.y);
    } else {
      oss << "|bl_x_m=nan|bl_y_m=nan|br_x_m=nan|br_y_m=nan";
    }

    oss << "|heading_valid=" << (tb.target_heading_valid ? 1 : 0)
        << "|heading_deg="
        << FormatLogFloat(tb.target_heading_valid
                              ? tb.target_heading_deg
                              : std::numeric_limits<double>::quiet_NaN());
  }

  return oss.str();
}

ObjectLogSummary BuildObjectLogSummary(const std::vector<TrackingBox>& objects,
                                       bool include_world_details) {
  ObjectLogSummary summary;
  for (const auto& tb : objects) {
    IncrementObjectCount(&summary, tb.class_id);
  }
  summary.object_summary = BuildObjectStateSummary(objects, include_world_details);
  return summary;
}

struct AccObjectLogSummary {
  int candidate_count = 0;
  int follow_count = 0;
  int lead_count = 0;
  int remaining_count = 0;
  bool target_box_valid = false;
  int target_box_x_px = 0;
  int target_box_y_px = 0;
  int target_box_w_px = 0;
  int target_box_h_px = 0;
  double target_bottom_center_u_px = 0.0;
  double target_bottom_center_v_px = 0.0;
  std::string object_state_summary;
};

AccObjectLogSummary BuildAccObjectLogSummary(const std::vector<TrackingBox>& world_result,
                                             const acc::AccCommand& acc_cmd) {
  AccObjectLogSummary summary;
  std::ostringstream oss;
  bool first = true;

  for (const auto& tb : world_result) {
    if (!(tb.class_id == 1 || tb.class_id == 2 || tb.class_id == 3)) {
      continue;
    }

    const acc::AccTrackedObjectState state = acc::ClassifyAccTrackedObjectState(acc_cmd, tb.id);
    switch (state) {
      case acc::AccTrackedObjectState::Candidate: summary.candidate_count += 1; break;
      case acc::AccTrackedObjectState::Lead: summary.lead_count += 1; break;
      case acc::AccTrackedObjectState::FollowingLead: summary.follow_count += 1; break;
      default: summary.remaining_count += 1; break;
    }

    cv::Point2f ground_xy;
    const bool ground_valid = acc::TryGetGroundBottomCenterXY(tb, ground_xy);
    const double bottom_center_u = tb.box.x + 0.5 * tb.box.width;
    const double bottom_center_v = tb.box.y + tb.box.height;

    if (!first) oss << ';';
    first = false;
    oss << "state=" << acc::AccTrackedObjectStateName(state)
        << "|id=" << tb.id
        << "|cls=" << tb.class_id
        << "|score=" << FormatLogFloat(tb.score)
        << "|x_m=" << FormatLogFloat(ground_valid ? ground_xy.x : std::numeric_limits<double>::quiet_NaN())
        << "|y_m=" << FormatLogFloat(ground_valid ? ground_xy.y : std::numeric_limits<double>::quiet_NaN())
        << "|u_px=" << FormatLogFloat(bottom_center_u)
        << "|v_px=" << FormatLogFloat(bottom_center_v)
        << "|box_x=" << tb.box.x
        << "|box_y=" << tb.box.y
        << "|box_w=" << tb.box.width
        << "|box_h=" << tb.box.height;

    if (tb.id == acc_cmd.target_id) {
      summary.target_box_valid = true;
      summary.target_box_x_px = tb.box.x;
      summary.target_box_y_px = tb.box.y;
      summary.target_box_w_px = tb.box.width;
      summary.target_box_h_px = tb.box.height;
      summary.target_bottom_center_u_px = bottom_center_u;
      summary.target_bottom_center_v_px = bottom_center_v;
    }
  }

  summary.object_state_summary = oss.str();
  return summary;
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

  log_frame->can_last_rx_sync_ns = can_state->can_last_rx_sync_ns;
  log_frame->can_powertrain_rx_sync_ns = can_state->can_powertrain_rx_sync_ns;
  log_frame->can_speed_rx_sync_ns = can_state->can_speed_rx_sync_ns;
  log_frame->can_yaw_rx_sync_ns = can_state->can_yaw_rx_sync_ns;
  log_frame->can_steer_rx_sync_ns = can_state->can_steer_rx_sync_ns;
  log_frame->can_steering_torque_rx_sync_ns = can_state->can_steering_torque_rx_sync_ns;
  log_frame->can_turn_signal_rx_sync_ns = can_state->can_turn_signal_rx_sync_ns;
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

  const unsigned char gear_ch = static_cast<unsigned char>(can_state->gear);
  log_frame->can_gear_text =
      std::isprint(gear_ch) ? std::string(1, static_cast<char>(gear_ch)) : "unknown";

  const unsigned char turn_signal_ch = static_cast<unsigned char>(can_state->turningSignal);
  log_frame->can_turn_signal_text =
      std::isprint(turn_signal_ch) ? std::string(1, static_cast<char>(turn_signal_ch)) : "unknown";
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

  const ObjectLogSummary tracking_summary =
      snapshot.tracking_result != nullptr
          ? BuildObjectLogSummary(*snapshot.tracking_result, false)
          : ObjectLogSummary{};
  const ObjectLogSummary world_summary = BuildObjectLogSummary(*snapshot.world_result, true);

  ResearchLogFrame log_frame;
  log_frame.frame_index = snapshot.frame_index;
  log_frame.frame_sync_ns = snapshot.frame_sync_ns;
  log_frame.frame_hw_ns = snapshot.frame_hw_ns;
  log_frame.cmd_sync_ns = snapshot.cmd_sync_ns;
  log_frame.can_steer_tx_sync_ns = TimeSyncGetCanSteerTxNs();
  log_frame.can_brake_tx_sync_ns = TimeSyncGetCanBrakeTxNs();
  log_frame.latency_frame_to_cmd_ms = TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, snapshot.frame_sync_ns);
  log_frame.latency_cmd_to_can_steer_tx_ms =
      TimeDeltaMsOrNaN(log_frame.can_steer_tx_sync_ns, snapshot.cmd_sync_ns);
  log_frame.latency_cmd_to_can_brake_tx_ms =
      TimeDeltaMsOrNaN(log_frame.can_brake_tx_sync_ns, snapshot.cmd_sync_ns);
  log_frame.latency_frame_to_can_steer_tx_ms =
      TimeDeltaMsOrNaN(log_frame.can_steer_tx_sync_ns, snapshot.frame_sync_ns);
  log_frame.latency_frame_to_can_brake_tx_ms =
      TimeDeltaMsOrNaN(log_frame.can_brake_tx_sync_ns, snapshot.frame_sync_ns);

  log_frame.dt_s = snapshot.dt_s;
  log_frame.ego_speed_kmh = snapshot.ego_speed_kmh;
  log_frame.perf_fps = snapshot.perf_fps;
  log_frame.perf_total_ms = snapshot.perf_total_ms;
  log_frame.perf_input_ms = snapshot.perf_input_ms;
  log_frame.perf_inference_ms = snapshot.perf_inference_ms;
  log_frame.perf_geometry_ms = snapshot.perf_geometry_ms;
  log_frame.perf_acc_scope_ms = snapshot.perf_acc_scope_ms;
  log_frame.perf_acc_ms = snapshot.perf_acc_ms;
  log_frame.perf_lka_ms = snapshot.perf_lka_ms;
  log_frame.perf_stability_ms = snapshot.perf_stability_ms;
  log_frame.perf_control_total_ms = snapshot.perf_control_total_ms;
  log_frame.perf_behavior_ms = snapshot.perf_behavior_ms;
  log_frame.perf_collision_ms = snapshot.perf_collision_ms;
  log_frame.perf_overlay_ms = snapshot.perf_overlay_ms;

  log_frame.cmd_speed_kmh = snapshot.target_speed_kmh;
  log_frame.cmd_steer_deg = snapshot.vehicle_cmd->steer_deg;
  log_frame.cmd_brake_0_10 = snapshot.vehicle_cmd->brake_0_10;
  log_frame.lka_steer_deg_raw = snapshot.vehicle_cmd->lka_steer_deg_raw;
  log_frame.lka_reference_valid = snapshot.lka_reference_valid;
  log_frame.lka_p_curve = snapshot.lka_p_curve;
  log_frame.lka_ey_m = snapshot.lka_ey_m;
  log_frame.lka_epsi_rad = snapshot.lka_epsi_rad;
  log_frame.lka_mean_kappa_m_inv = snapshot.lka_mean_kappa_m_inv;
  log_frame.lka_std_kappa_m_inv = snapshot.lka_std_kappa_m_inv;
  log_frame.lka_current_x_m = snapshot.lka_current_x_m;
  log_frame.lka_current_y_m = snapshot.lka_current_y_m;
  log_frame.lka_current_image_valid = snapshot.lka_current_image_valid;
  log_frame.lka_current_u_px = snapshot.lka_current_u_px;
  log_frame.lka_current_v_px = snapshot.lka_current_v_px;
  log_frame.lka_target_x_m = snapshot.lka_target_x_m;
  log_frame.lka_target_y_m = snapshot.lka_target_y_m;
  log_frame.lka_target_image_valid = snapshot.lka_target_image_valid;
  log_frame.lka_target_u_px = snapshot.lka_target_u_px;
  log_frame.lka_target_v_px = snapshot.lka_target_v_px;
  log_frame.tracking_object_count = tracking_summary.total_count;
  log_frame.tracking_lane_count = tracking_summary.lane_count;
  log_frame.tracking_car_count = tracking_summary.car_count;
  log_frame.tracking_rider_count = tracking_summary.rider_count;
  log_frame.tracking_person_count = tracking_summary.person_count;
  log_frame.tracking_light_count = tracking_summary.light_count;
  log_frame.tracking_signc_count = tracking_summary.signc_count;
  log_frame.tracking_signt_count = tracking_summary.signt_count;
  log_frame.tracking_other_count = tracking_summary.other_count;
  log_frame.tracking_object_summary = tracking_summary.object_summary;

  const auto& acc_cmd = snapshot.vehicle_cmd->acc_cmd;
  const AccObjectLogSummary acc_summary = BuildAccObjectLogSummary(*snapshot.world_result, acc_cmd);
  const acc::AccTrackedObjectState lead_state = acc::ClassifyAccTrackedObjectState(acc_cmd, acc_cmd.target_id);

  log_frame.acc_has_lead = acc_cmd.has_lead;
  log_frame.acc_lead_following_active = acc_cmd.lead_following_active;
  log_frame.acc_lead_state_code = acc::AccTrackedObjectStateCode(lead_state);
  log_frame.acc_lead_state_text = acc::AccTrackedObjectStateName(lead_state);
  log_frame.acc_candidate_count = acc_summary.candidate_count;
  log_frame.acc_follow_count = acc_summary.follow_count;
  log_frame.acc_lead_count = acc_summary.lead_count;
  log_frame.acc_remaining_count = acc_summary.remaining_count;
  log_frame.acc_target_id = acc_cmd.target_id;
  log_frame.acc_target_speed_kmh = acc_cmd.TargetSpeedKmh;
  log_frame.acc_target_distance_m = snapshot.target_distance_m;
  log_frame.acc_target_lateral_m = acc_cmd.lead_lateral_m;
  log_frame.acc_target_relative_speed_mps = acc_cmd.relative_speed_mps;
  log_frame.acc_target_score = acc_cmd.TargetScore;
  log_frame.acc_target_dist_std_m = acc_cmd.TargetDistStd;
  log_frame.acc_target_rel_speed_std_mps = acc_cmd.RelSpeedStd;
  log_frame.acc_target_ttc_s = snapshot.target_ttc_s;
  log_frame.acc_target_ttc_std_s = acc_cmd.TargetTTCStd;
  log_frame.acc_target_box_valid = acc_summary.target_box_valid;
  log_frame.acc_target_box_x_px = acc_summary.target_box_x_px;
  log_frame.acc_target_box_y_px = acc_summary.target_box_y_px;
  log_frame.acc_target_box_w_px = acc_summary.target_box_w_px;
  log_frame.acc_target_box_h_px = acc_summary.target_box_h_px;
  log_frame.acc_target_bottom_center_u_px = acc_summary.target_bottom_center_u_px;
  log_frame.acc_target_bottom_center_v_px = acc_summary.target_bottom_center_v_px;
  log_frame.acc_longitudinal_phase_code = acc::AccLongitudinalPhaseCode(acc_cmd.longitudinal_phase);
  log_frame.acc_longitudinal_phase_text = acc::AccLongitudinalPhaseName(acc_cmd.longitudinal_phase);
  log_frame.acc_control_ego_speed_kmh = acc_cmd.ego_speed_kmh;
  log_frame.acc_control_cruise_speed_kmh = acc_cmd.cruise_speed_kmh;
  log_frame.acc_control_speed_cmd_kmh = acc_cmd.speed_kmh;
  log_frame.acc_control_brake_0_10 = acc_cmd.brake_0_10;
  log_frame.acc_control_accel_cmd_mps2 = acc_cmd.accel_cmd_mps2;
  log_frame.acc_control_free_accel_nom_mps2 = acc_cmd.free_accel_nom_mps2;
  log_frame.acc_control_free_accel_limited_mps2 = acc_cmd.free_accel_limited_mps2;
  log_frame.acc_object_state_summary = acc_summary.object_state_summary;

  log_frame.collision_warning = snapshot.collision_output->warning;
  log_frame.collision_threat_id = snapshot.collision_output->threat_id;
  log_frame.collision_threat_ttc_s = snapshot.collision_output->threat_ttc_s;
  log_frame.collision_threat_dist_now_m = snapshot.collision_output->threat_dist_now_m;
  log_frame.collision_threat_min_dist_m = snapshot.collision_output->threat_min_dist_m;
  log_frame.collision_threat_approach_speed_mps = snapshot.collision_output->threat_approach_speed_mps;
  log_frame.collision_threat_pos_x_m = snapshot.collision_output->threat_pos.x;
  log_frame.collision_threat_pos_y_m = snapshot.collision_output->threat_pos.y;

  log_frame.world_object_count = world_summary.total_count;
  log_frame.world_lane_count = world_summary.lane_count;
  log_frame.world_car_count = world_summary.car_count;
  log_frame.world_person_count = world_summary.person_count;
  log_frame.world_rider_count = world_summary.rider_count;
  log_frame.world_light_count = world_summary.light_count;
  log_frame.world_signc_count = world_summary.signc_count;
  log_frame.world_signt_count = world_summary.signt_count;
  log_frame.world_other_count = world_summary.other_count;
  log_frame.world_object_summary = world_summary.object_summary;

  FillCanState(&log_frame, snapshot.can_valid, snapshot.can_state);
  log_frame.can_last_rx_age_at_cmd_ms =
      TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, log_frame.can_last_rx_sync_ns);
  log_frame.can_powertrain_age_at_cmd_ms =
      TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, log_frame.can_powertrain_rx_sync_ns);
  log_frame.can_speed_age_at_cmd_ms =
      TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, log_frame.can_speed_rx_sync_ns);
  log_frame.can_yaw_age_at_cmd_ms =
      TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, log_frame.can_yaw_rx_sync_ns);
  log_frame.can_steer_age_at_cmd_ms =
      TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, log_frame.can_steer_rx_sync_ns);
  log_frame.can_steering_torque_age_at_cmd_ms =
      TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, log_frame.can_steering_torque_rx_sync_ns);
  log_frame.can_turn_signal_age_at_cmd_ms =
      TimeDeltaMsOrNaN(snapshot.cmd_sync_ns, log_frame.can_turn_signal_rx_sync_ns);
  research_logger_.LogFrame(log_frame);
}

void RuntimeLogManager::Stop() {
  research_logger_.Stop();
  ablation_logger_.Stop();
}

}  // namespace adas_log
