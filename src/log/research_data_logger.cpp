#include "research_data_logger.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

bool IsFinite(double v) {
  return std::isfinite(v);
}

std::string TimestampStringNow() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);

  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif

  char buf[32] = {0};
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
  return std::string(buf);
}

std::string ParentDirOf(const std::string& p) {
  std::filesystem::path path(p);
  return path.has_parent_path() ? path.parent_path().string() : std::string();
}

}  // namespace

ResearchDataLogger::ResearchDataLogger(const ResearchLogOptions& options)
    : options_(options) {}

ResearchDataLogger::~ResearchDataLogger() {
  Stop();
}

bool ResearchDataLogger::ParseEnvBool(const char* value, bool default_value) {
  if (!value) return default_value;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  return default_value;
}

double ResearchDataLogger::Clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

double ResearchDataLogger::DegToRad(double deg) {
  return deg * (3.14159265358979323846 / 180.0);
}

double ResearchDataLogger::WrapPi(double rad) {
  constexpr double kPi = 3.14159265358979323846;
  while (rad > kPi) rad -= 2.0 * kPi;
  while (rad < -kPi) rad += 2.0 * kPi;
  return rad;
}

double ResearchDataLogger::FiniteOrNaN(double value) {
  if (IsFinite(value)) return value;
  return std::numeric_limits<double>::quiet_NaN();
}

std::string ResearchDataLogger::ResolveOutputPath() const {
  const char* env_path = std::getenv("ADAS_RESEARCH_LOG_PATH");
  if (env_path && *env_path) return std::string(env_path);

  const char* env_dir = std::getenv("ADAS_RESEARCH_LOG_DIR");
  const std::string out_dir = (env_dir && *env_dir) ? std::string(env_dir) : options_.output_dir;

  std::ostringstream oss;
  oss << out_dir << "/research_drive_" << TimestampStringNow() << ".csv";
  return oss.str();
}

bool ResearchDataLogger::Start(std::string* out_error) {
  const char* env_enable = std::getenv("ADAS_RESEARCH_LOG_ENABLE");
  options_.enabled = ParseEnvBool(env_enable, options_.enabled);

  if (!options_.enabled) {
    running_ = false;
    return true;
  }

  output_path_ = options_.output_path.empty() ? ResolveOutputPath() : options_.output_path;

  const std::string parent_dir = ParentDirOf(output_path_);
  if (!parent_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent_dir, ec);
    if (ec) {
      if (out_error) {
        *out_error = "failed to create output directory: " + parent_dir + " (" + ec.message() + ")";
      }
      return false;
    }
  }

  out_.open(output_path_, std::ios::out | std::ios::trunc);
  if (!out_.is_open()) {
    if (out_error) {
      *out_error = "failed to open log file: " + output_path_;
    }
    return false;
  }

  out_ << std::fixed << std::setprecision(6);
  WriteHeader();

  flush_counter_ = 0;
  route_x_m_ = 0.0;
  route_y_m_ = 0.0;
  route_heading_rad_ = 0.0;
  route_distance_m_ = 0.0;
  route_source_ = 0;
  summary_ = Summary{};

  start_steady_ = std::chrono::steady_clock::now();
  running_ = true;
  return true;
}

void ResearchDataLogger::WriteHeader() {
  out_ << "unix_ms"
       << ",elapsed_s"
       << ",frame_idx"
       << ",frame_sync_ns"
       << ",frame_hw_ns"
       << ",cmd_sync_ns"
       << ",can_steer_tx_sync_ns"
       << ",can_brake_tx_sync_ns"
       << ",dt_s"
       << ",ego_speed_kmh"
       << ",perf_fps"
       << ",perf_total_ms"
       << ",perf_input_ms"
       << ",perf_inference_ms"
       << ",perf_geometry_ms"
       << ",perf_acc_scope_ms"
       << ",perf_acc_ms"
       << ",perf_lka_ms"
       << ",perf_stability_ms"
       << ",perf_control_total_ms"
       << ",perf_behavior_ms"
       << ",perf_collision_ms"
       << ",perf_overlay_ms"
       << ",cmd_speed_kmh"
       << ",cmd_steer_deg"
       << ",cmd_brake_0_10"
       << ",lka_steer_deg_raw"
       << ",lka_reference_valid"
       << ",lka_p_curve"
       << ",lka_ey_m"
       << ",lka_epsi_rad"
       << ",lka_mean_kappa_m_inv"
       << ",lka_std_kappa_m_inv"
       << ",lka_current_x_m"
       << ",lka_current_y_m"
       << ",lka_current_image_valid"
       << ",lka_current_u_px"
       << ",lka_current_v_px"
       << ",lka_target_x_m"
       << ",lka_target_y_m"
       << ",lka_target_image_valid"
       << ",lka_target_u_px"
       << ",lka_target_v_px"
       << ",acc_has_lead"
       << ",acc_lead_following_active"
       << ",acc_lead_state_code"
       << ",acc_lead_state_text"
       << ",acc_candidate_count"
       << ",acc_follow_count"
       << ",acc_lead_count"
       << ",acc_remaining_count"
       << ",acc_target_id"
       << ",acc_target_speed_kmh"
       << ",acc_target_distance_m"
       << ",acc_target_lateral_m"
       << ",acc_target_relative_speed_mps"
       << ",acc_target_score"
       << ",acc_target_dist_std_m"
       << ",acc_target_rel_speed_std_mps"
       << ",acc_target_ttc_s"
       << ",acc_target_ttc_std_s"
       << ",acc_target_box_valid"
       << ",acc_target_box_x_px"
       << ",acc_target_box_y_px"
       << ",acc_target_box_w_px"
       << ",acc_target_box_h_px"
       << ",acc_target_bottom_center_u_px"
       << ",acc_target_bottom_center_v_px"
       << ",acc_longitudinal_phase_code"
       << ",acc_longitudinal_phase_text"
       << ",acc_control_ego_speed_kmh"
       << ",acc_control_cruise_speed_kmh"
       << ",acc_control_speed_cmd_kmh"
       << ",acc_control_brake_0_10"
       << ",acc_control_accel_cmd_mps2"
       << ",acc_control_free_accel_nom_mps2"
       << ",acc_control_free_accel_limited_mps2"
       << ",acc_object_state_summary"
       << ",collision_warning"
       << ",collision_threat_id"
       << ",collision_threat_ttc_s"
       << ",collision_threat_dist_now_m"
       << ",collision_threat_min_dist_m"
       << ",collision_threat_approach_speed_mps"
       << ",collision_threat_pos_x_m"
       << ",collision_threat_pos_y_m"
       << ",world_object_count"
       << ",world_car_count"
       << ",world_person_count"
       << ",world_rider_count"
       << ",can_valid"
       << ",can_speed_kmh"
       << ",can_speed_raw_kmh"
       << ",can_steer_deg"
       << ",can_yaw_deg_s"
       << ",can_theta_deg"
       << ",can_lat_accel_mps2"
       << ",can_long_accel_mps2"
       << ",can_steering_torque_nm"
       << ",can_meterage_m"
       << ",can_throttle"
       << ",can_gear"
       << ",can_turn_signal"
       << ",route_source"
       << ",route_heading_rad"
       << ",route_x_m"
       << ",route_y_m"
       << ",route_distance_m"
       << ",tracking_object_count"
       << ",tracking_lane_count"
       << ",tracking_car_count"
       << ",tracking_rider_count"
       << ",tracking_person_count"
       << ",tracking_light_count"
       << ",tracking_signc_count"
       << ",tracking_signt_count"
       << ",tracking_other_count"
       << ",tracking_object_summary"
       << ",world_lane_count"
       << ",world_light_count"
       << ",world_signc_count"
       << ",world_signt_count"
       << ",world_other_count"
       << ",world_object_summary"
       << ",can_last_rx_sync_ns"
       << ",can_powertrain_rx_sync_ns"
       << ",can_speed_rx_sync_ns"
       << ",can_yaw_rx_sync_ns"
       << ",can_steer_rx_sync_ns"
       << ",can_steering_torque_rx_sync_ns"
       << ",can_turn_signal_rx_sync_ns"
       << ",latency_frame_to_cmd_ms"
       << ",latency_cmd_to_can_steer_tx_ms"
       << ",latency_cmd_to_can_brake_tx_ms"
       << ",latency_frame_to_can_steer_tx_ms"
       << ",latency_frame_to_can_brake_tx_ms"
       << ",can_last_rx_age_at_cmd_ms"
       << ",can_powertrain_age_at_cmd_ms"
       << ",can_speed_age_at_cmd_ms"
       << ",can_yaw_age_at_cmd_ms"
       << ",can_steer_age_at_cmd_ms"
       << ",can_steering_torque_age_at_cmd_ms"
       << ",can_turn_signal_age_at_cmd_ms"
       << ",can_gear_text"
       << ",can_turn_signal_text"
       << '\n';
}

void ResearchDataLogger::UpdatePose(const ResearchLogFrame& frame, double dt_s) {
  const double speed_kmh = (frame.can_valid && IsFinite(frame.can_speed_kmh))
                             ? std::max(0.0, frame.can_speed_kmh)
                             : std::max(0.0, frame.ego_speed_kmh);
  const double speed_mps = speed_kmh / 3.6;

  if (frame.can_valid && IsFinite(frame.can_theta_deg)) {
    route_heading_rad_ = DegToRad(frame.can_theta_deg);
    route_source_ = 2;
  } else if (frame.can_valid && IsFinite(frame.can_yaw_deg_s)) {
    route_heading_rad_ += DegToRad(frame.can_yaw_deg_s) * dt_s;
    route_source_ = 1;
  } else {
    const double steering_ratio = std::max(1e-3, options_.steering_ratio);
    const double wheelbase_m = std::max(1e-3, options_.wheelbase_m);
    const double road_wheel_deg = frame.cmd_steer_deg / steering_ratio;
    const double delta_rad = DegToRad(road_wheel_deg);
    const double yaw_rate_rps = speed_mps * std::tan(delta_rad) / wheelbase_m;
    route_heading_rad_ += yaw_rate_rps * dt_s;
    route_source_ = 0;
  }

  route_heading_rad_ = WrapPi(route_heading_rad_);

  const double ds = speed_mps * dt_s;
  route_x_m_ += ds * std::cos(route_heading_rad_);
  route_y_m_ += ds * std::sin(route_heading_rad_);
  route_distance_m_ += ds;
}

void ResearchDataLogger::UpdateSummary(const ResearchLogFrame& frame, double elapsed_s) {
  summary_.sample_count += 1;
  if (frame.collision_warning) summary_.warning_count += 1;
  if (frame.cmd_brake_0_10 > 1e-6) summary_.braking_count += 1;

  const double speed_kmh = std::max(0.0, frame.ego_speed_kmh);
  summary_.sum_speed_kmh += speed_kmh;
  summary_.max_speed_kmh = std::max(summary_.max_speed_kmh, speed_kmh);

  const double abs_steer = std::abs(frame.cmd_steer_deg);
  summary_.sum_abs_steer_deg += abs_steer;
  summary_.max_abs_steer_deg = std::max(summary_.max_abs_steer_deg, abs_steer);

  if (IsFinite(frame.acc_target_ttc_s) && frame.acc_target_ttc_s >= 0.0) {
    summary_.min_acc_ttc_s = std::min(summary_.min_acc_ttc_s, frame.acc_target_ttc_s);
  }

  if (IsFinite(frame.collision_threat_ttc_s) && frame.collision_threat_ttc_s >= 0.0) {
    summary_.min_collision_ttc_s = std::min(summary_.min_collision_ttc_s, frame.collision_threat_ttc_s);
  }

  if (!summary_.has_first_sample) {
    summary_.first_elapsed_s = elapsed_s;
    summary_.has_first_sample = true;
  }
  summary_.last_elapsed_s = elapsed_s;
}

void ResearchDataLogger::LogFrame(const ResearchLogFrame& frame) {
  if (!running_) return;

  const auto now_wall = std::chrono::system_clock::now();
  const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now_wall.time_since_epoch()).count();
  const double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start_steady_).count();

  const double dt_s = Clamp(frame.dt_s, 0.001, 1.0);
  UpdatePose(frame, dt_s);
  UpdateSummary(frame, elapsed_s);

  out_ << unix_ms
       << ',' << elapsed_s
       << ',' << frame.frame_index
       << ',' << frame.frame_sync_ns
       << ',' << frame.frame_hw_ns
       << ',' << frame.cmd_sync_ns
       << ',' << frame.can_steer_tx_sync_ns
       << ',' << frame.can_brake_tx_sync_ns
       << ',' << FiniteOrNaN(dt_s)
       << ',' << FiniteOrNaN(frame.ego_speed_kmh)
       << ',' << FiniteOrNaN(frame.perf_fps)
       << ',' << FiniteOrNaN(frame.perf_total_ms)
       << ',' << FiniteOrNaN(frame.perf_input_ms)
       << ',' << FiniteOrNaN(frame.perf_inference_ms)
       << ',' << FiniteOrNaN(frame.perf_geometry_ms)
       << ',' << FiniteOrNaN(frame.perf_acc_scope_ms)
       << ',' << FiniteOrNaN(frame.perf_acc_ms)
       << ',' << FiniteOrNaN(frame.perf_lka_ms)
       << ',' << FiniteOrNaN(frame.perf_stability_ms)
       << ',' << FiniteOrNaN(frame.perf_control_total_ms)
       << ',' << FiniteOrNaN(frame.perf_behavior_ms)
       << ',' << FiniteOrNaN(frame.perf_collision_ms)
       << ',' << FiniteOrNaN(frame.perf_overlay_ms)
       << ',' << FiniteOrNaN(frame.cmd_speed_kmh)
       << ',' << FiniteOrNaN(frame.cmd_steer_deg)
       << ',' << FiniteOrNaN(frame.cmd_brake_0_10)
       << ',' << FiniteOrNaN(frame.lka_steer_deg_raw)
       << ',' << (frame.lka_reference_valid ? 1 : 0)
       << ',' << FiniteOrNaN(frame.lka_p_curve)
       << ',' << FiniteOrNaN(frame.lka_ey_m)
       << ',' << FiniteOrNaN(frame.lka_epsi_rad)
       << ',' << FiniteOrNaN(frame.lka_mean_kappa_m_inv)
       << ',' << FiniteOrNaN(frame.lka_std_kappa_m_inv)
       << ',' << FiniteOrNaN(frame.lka_current_x_m)
       << ',' << FiniteOrNaN(frame.lka_current_y_m)
       << ',' << (frame.lka_current_image_valid ? 1 : 0)
       << ',' << FiniteOrNaN(frame.lka_current_u_px)
       << ',' << FiniteOrNaN(frame.lka_current_v_px)
       << ',' << FiniteOrNaN(frame.lka_target_x_m)
       << ',' << FiniteOrNaN(frame.lka_target_y_m)
       << ',' << (frame.lka_target_image_valid ? 1 : 0)
       << ',' << FiniteOrNaN(frame.lka_target_u_px)
       << ',' << FiniteOrNaN(frame.lka_target_v_px)
       << ',' << (frame.acc_has_lead ? 1 : 0)
       << ',' << (frame.acc_lead_following_active ? 1 : 0)
       << ',' << frame.acc_lead_state_code
       << ',' << frame.acc_lead_state_text
       << ',' << frame.acc_candidate_count
       << ',' << frame.acc_follow_count
       << ',' << frame.acc_lead_count
       << ',' << frame.acc_remaining_count
       << ',' << frame.acc_target_id
       << ',' << FiniteOrNaN(frame.acc_target_speed_kmh)
       << ',' << FiniteOrNaN(frame.acc_target_distance_m)
       << ',' << FiniteOrNaN(frame.acc_target_lateral_m)
       << ',' << FiniteOrNaN(frame.acc_target_relative_speed_mps)
       << ',' << FiniteOrNaN(frame.acc_target_score)
       << ',' << FiniteOrNaN(frame.acc_target_dist_std_m)
       << ',' << FiniteOrNaN(frame.acc_target_rel_speed_std_mps)
       << ',' << FiniteOrNaN(frame.acc_target_ttc_s)
       << ',' << FiniteOrNaN(frame.acc_target_ttc_std_s)
       << ',' << (frame.acc_target_box_valid ? 1 : 0)
       << ',' << frame.acc_target_box_x_px
       << ',' << frame.acc_target_box_y_px
       << ',' << frame.acc_target_box_w_px
       << ',' << frame.acc_target_box_h_px
       << ',' << FiniteOrNaN(frame.acc_target_bottom_center_u_px)
       << ',' << FiniteOrNaN(frame.acc_target_bottom_center_v_px)
       << ',' << frame.acc_longitudinal_phase_code
       << ',' << frame.acc_longitudinal_phase_text
       << ',' << FiniteOrNaN(frame.acc_control_ego_speed_kmh)
       << ',' << FiniteOrNaN(frame.acc_control_cruise_speed_kmh)
       << ',' << FiniteOrNaN(frame.acc_control_speed_cmd_kmh)
       << ',' << FiniteOrNaN(frame.acc_control_brake_0_10)
       << ',' << FiniteOrNaN(frame.acc_control_accel_cmd_mps2)
       << ',' << FiniteOrNaN(frame.acc_control_free_accel_nom_mps2)
       << ',' << FiniteOrNaN(frame.acc_control_free_accel_limited_mps2)
       << ',' << frame.acc_object_state_summary
       << ',' << (frame.collision_warning ? 1 : 0)
       << ',' << frame.collision_threat_id
       << ',' << FiniteOrNaN(frame.collision_threat_ttc_s)
       << ',' << FiniteOrNaN(frame.collision_threat_dist_now_m)
       << ',' << FiniteOrNaN(frame.collision_threat_min_dist_m)
       << ',' << FiniteOrNaN(frame.collision_threat_approach_speed_mps)
       << ',' << FiniteOrNaN(frame.collision_threat_pos_x_m)
       << ',' << FiniteOrNaN(frame.collision_threat_pos_y_m)
       << ',' << frame.world_object_count
       << ',' << frame.world_car_count
       << ',' << frame.world_person_count
       << ',' << frame.world_rider_count
       << ',' << (frame.can_valid ? 1 : 0)
       << ',' << FiniteOrNaN(frame.can_speed_kmh)
       << ',' << FiniteOrNaN(frame.can_speed_raw_kmh)
       << ',' << FiniteOrNaN(frame.can_steer_deg)
       << ',' << FiniteOrNaN(frame.can_yaw_deg_s)
       << ',' << FiniteOrNaN(frame.can_theta_deg)
       << ',' << FiniteOrNaN(frame.can_lat_accel_mps2)
       << ',' << FiniteOrNaN(frame.can_long_accel_mps2)
       << ',' << FiniteOrNaN(frame.can_steering_torque_nm)
       << ',' << FiniteOrNaN(frame.can_meterage_m)
       << ',' << frame.can_throttle
       << ',' << frame.can_gear
       << ',' << frame.can_turn_signal
       << ',' << route_source_
       << ',' << FiniteOrNaN(route_heading_rad_)
       << ',' << FiniteOrNaN(route_x_m_)
       << ',' << FiniteOrNaN(route_y_m_)
       << ',' << FiniteOrNaN(route_distance_m_)
       << ',' << frame.tracking_object_count
       << ',' << frame.tracking_lane_count
       << ',' << frame.tracking_car_count
       << ',' << frame.tracking_rider_count
       << ',' << frame.tracking_person_count
       << ',' << frame.tracking_light_count
       << ',' << frame.tracking_signc_count
       << ',' << frame.tracking_signt_count
       << ',' << frame.tracking_other_count
       << ',' << frame.tracking_object_summary
       << ',' << frame.world_lane_count
       << ',' << frame.world_light_count
       << ',' << frame.world_signc_count
       << ',' << frame.world_signt_count
       << ',' << frame.world_other_count
       << ',' << frame.world_object_summary
       << ',' << frame.can_last_rx_sync_ns
       << ',' << frame.can_powertrain_rx_sync_ns
       << ',' << frame.can_speed_rx_sync_ns
       << ',' << frame.can_yaw_rx_sync_ns
       << ',' << frame.can_steer_rx_sync_ns
       << ',' << frame.can_steering_torque_rx_sync_ns
       << ',' << frame.can_turn_signal_rx_sync_ns
       << ',' << FiniteOrNaN(frame.latency_frame_to_cmd_ms)
       << ',' << FiniteOrNaN(frame.latency_cmd_to_can_steer_tx_ms)
       << ',' << FiniteOrNaN(frame.latency_cmd_to_can_brake_tx_ms)
       << ',' << FiniteOrNaN(frame.latency_frame_to_can_steer_tx_ms)
       << ',' << FiniteOrNaN(frame.latency_frame_to_can_brake_tx_ms)
       << ',' << FiniteOrNaN(frame.can_last_rx_age_at_cmd_ms)
       << ',' << FiniteOrNaN(frame.can_powertrain_age_at_cmd_ms)
       << ',' << FiniteOrNaN(frame.can_speed_age_at_cmd_ms)
       << ',' << FiniteOrNaN(frame.can_yaw_age_at_cmd_ms)
       << ',' << FiniteOrNaN(frame.can_steer_age_at_cmd_ms)
       << ',' << FiniteOrNaN(frame.can_steering_torque_age_at_cmd_ms)
       << ',' << FiniteOrNaN(frame.can_turn_signal_age_at_cmd_ms)
       << ',' << frame.can_gear_text
       << ',' << frame.can_turn_signal_text
       << '\n';

  flush_counter_ += 1;
  if (flush_counter_ >= static_cast<uint64_t>(std::max(1, options_.flush_every_n))) {
    out_.flush();
    flush_counter_ = 0;
  }
}

void ResearchDataLogger::WriteSummaryFile() {
  if (output_path_.empty()) return;

  std::ofstream summary(output_path_ + ".summary.txt", std::ios::out | std::ios::trunc);
  if (!summary.is_open()) return;

  summary << std::fixed << std::setprecision(6);

  const double duration_s = summary_.has_first_sample
      ? std::max(0.0, summary_.last_elapsed_s - summary_.first_elapsed_s)
      : 0.0;

  const double avg_speed_kmh = (summary_.sample_count > 0)
      ? (summary_.sum_speed_kmh / static_cast<double>(summary_.sample_count))
      : 0.0;

  const double avg_abs_steer_deg = (summary_.sample_count > 0)
      ? (summary_.sum_abs_steer_deg / static_cast<double>(summary_.sample_count))
      : 0.0;

  const double warning_ratio = (summary_.sample_count > 0)
      ? (static_cast<double>(summary_.warning_count) / static_cast<double>(summary_.sample_count))
      : 0.0;

  const double braking_ratio = (summary_.sample_count > 0)
      ? (static_cast<double>(summary_.braking_count) / static_cast<double>(summary_.sample_count))
      : 0.0;

  summary << "output_csv=" << output_path_ << '\n';
  summary << "time_sync_source=" << options_.time_sync_source << '\n';
  summary << "time_sync_uses_ptp=" << (options_.time_sync_uses_ptp ? 1 : 0) << '\n';
  summary << "run_mode=" << options_.run_mode << '\n';
  summary << "input_source=" << options_.input_source << '\n';
  summary << "can_tx_master_enable=" << (options_.can_tx_master_enable ? 1 : 0) << '\n';
  summary << "can_throttle_enable=" << (options_.can_throttle_enable ? 1 : 0) << '\n';
  summary << "can_brake_enable=" << (options_.can_brake_enable ? 1 : 0) << '\n';
  summary << "can_longitudinal_enable=" << (options_.can_longitudinal_enable ? 1 : 0) << '\n';
  summary << "can_steering_enable=" << (options_.can_steering_enable ? 1 : 0) << '\n';
  summary << "samples=" << summary_.sample_count << '\n';
  summary << "duration_s=" << duration_s << '\n';
  summary << "route_distance_est_m=" << route_distance_m_ << '\n';
  summary << "route_end_x_m=" << route_x_m_ << '\n';
  summary << "route_end_y_m=" << route_y_m_ << '\n';
  summary << "avg_speed_kmh=" << avg_speed_kmh << '\n';
  summary << "max_speed_kmh=" << summary_.max_speed_kmh << '\n';
  summary << "avg_abs_steer_deg=" << avg_abs_steer_deg << '\n';
  summary << "max_abs_steer_deg=" << summary_.max_abs_steer_deg << '\n';
  summary << "collision_warning_ratio=" << warning_ratio << '\n';
  summary << "braking_ratio=" << braking_ratio << '\n';

  if (summary_.min_acc_ttc_s < 1e29) {
    summary << "min_acc_ttc_s=" << summary_.min_acc_ttc_s << '\n';
  } else {
    summary << "min_acc_ttc_s=nan\n";
  }

  if (summary_.min_collision_ttc_s < 1e29) {
    summary << "min_collision_ttc_s=" << summary_.min_collision_ttc_s << '\n';
  } else {
    summary << "min_collision_ttc_s=nan\n";
  }
}

void ResearchDataLogger::Stop() {
  if (!running_) return;

  WriteSummaryFile();

  if (out_.is_open()) {
    out_.flush();
    out_.close();
  }

  running_ = false;
}
