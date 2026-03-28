#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>

struct ResearchLogOptions {
  bool enabled = true;
  std::string output_path;
  std::string output_dir = "research_logs";
  double steering_ratio = 14.5;
  double wheelbase_m = 2.62;
  int flush_every_n = 30;
  bool time_sync_uses_ptp = false;
  std::string time_sync_source = "CLOCK_REALTIME";
};

struct ResearchLogFrame {
  uint64_t frame_index = 0;
  uint64_t frame_sync_ns = 0;
  uint64_t frame_hw_ns = 0;
  uint64_t cmd_sync_ns = 0;
  uint64_t can_steer_tx_sync_ns = 0;
  uint64_t can_brake_tx_sync_ns = 0;

  double dt_s = 0.0;
  double ego_speed_kmh = 0.0;
  double perf_fps = 0.0;
  double perf_total_ms = 0.0;
  double perf_input_ms = 0.0;
  double perf_inference_ms = 0.0;
  double perf_geometry_ms = 0.0;
  double perf_acc_scope_ms = 0.0;
  double perf_acc_ms = 0.0;
  double perf_lka_ms = 0.0;
  double perf_stability_ms = 0.0;
  double perf_control_total_ms = 0.0;
  double perf_behavior_ms = 0.0;
  double perf_collision_ms = 0.0;
  double perf_overlay_ms = 0.0;

  double cmd_speed_kmh = 0.0;
  double cmd_steer_deg = 0.0;
  double cmd_brake_0_10 = 0.0;
  double lka_steer_deg_raw = 0.0;
  bool lka_reference_valid = false;
  double lka_p_curve = 0.0;
  double lka_current_x_m = 0.0;
  double lka_current_y_m = 0.0;
  bool lka_current_image_valid = false;
  double lka_current_u_px = 0.0;
  double lka_current_v_px = 0.0;
  double lka_target_x_m = 0.0;
  double lka_target_y_m = 0.0;
  bool lka_target_image_valid = false;
  double lka_target_u_px = 0.0;
  double lka_target_v_px = 0.0;

  bool acc_has_lead = false;
  bool acc_lead_following_active = false;
  int acc_lead_state_code = 0;
  std::string acc_lead_state_text = "remaining";
  int acc_candidate_count = 0;
  int acc_follow_count = 0;
  int acc_lead_count = 0;
  int acc_remaining_count = 0;

  int acc_target_id = -1;
  double acc_target_speed_kmh = 0.0;
  double acc_target_distance_m = 0.0;
  double acc_target_lateral_m = 0.0;
  double acc_target_relative_speed_mps = 0.0;
  double acc_target_score = 0.0;
  double acc_target_dist_std_m = 0.0;
  double acc_target_rel_speed_std_mps = 0.0;
  double acc_target_ttc_s = 0.0;
  double acc_target_ttc_std_s = 0.0;
  bool acc_target_box_valid = false;
  int acc_target_box_x_px = 0;
  int acc_target_box_y_px = 0;
  int acc_target_box_w_px = 0;
  int acc_target_box_h_px = 0;
  double acc_target_bottom_center_u_px = 0.0;
  double acc_target_bottom_center_v_px = 0.0;

  int acc_longitudinal_phase_code = 0;
  std::string acc_longitudinal_phase_text = "idle";
  double acc_control_ego_speed_kmh = 0.0;
  double acc_control_cruise_speed_kmh = 0.0;
  double acc_control_speed_cmd_kmh = 0.0;
  double acc_control_brake_0_10 = 0.0;
  double acc_control_accel_cmd_mps2 = 0.0;
  double acc_control_free_accel_nom_mps2 = 0.0;
  double acc_control_free_accel_limited_mps2 = 0.0;
  std::string acc_object_state_summary;

  bool collision_warning = false;
  int collision_threat_id = -1;
  double collision_threat_ttc_s = 0.0;
  double collision_threat_dist_now_m = 0.0;
  double collision_threat_min_dist_m = 0.0;
  double collision_threat_approach_speed_mps = 0.0;
  double collision_threat_pos_x_m = 0.0;
  double collision_threat_pos_y_m = 0.0;

  int world_object_count = 0;
  int world_car_count = 0;
  int world_person_count = 0;
  int world_rider_count = 0;

  bool can_valid = false;
  double can_speed_kmh = 0.0;
  double can_speed_raw_kmh = 0.0;
  double can_steer_deg = 0.0;
  double can_yaw_deg_s = 0.0;
  double can_theta_deg = 0.0;
  double can_lat_accel_mps2 = 0.0;
  double can_long_accel_mps2 = 0.0;
  double can_steering_torque_nm = 0.0;
  double can_meterage_m = 0.0;
  int can_throttle = 0;
  int can_gear = 0;
  int can_turn_signal = 0;
};

class ResearchDataLogger {
public:
  explicit ResearchDataLogger(const ResearchLogOptions& options = {});
  ~ResearchDataLogger();

  bool Start(std::string* out_error = nullptr);
  void LogFrame(const ResearchLogFrame& frame);
  void Stop();

  bool IsEnabled() const { return options_.enabled; }
  bool IsRunning() const { return running_; }
  const std::string& OutputPath() const { return output_path_; }

private:
  struct Summary {
    uint64_t sample_count = 0;
    uint64_t warning_count = 0;
    uint64_t braking_count = 0;

    double sum_speed_kmh = 0.0;
    double max_speed_kmh = 0.0;

    double sum_abs_steer_deg = 0.0;
    double max_abs_steer_deg = 0.0;

    double min_acc_ttc_s = 1e30;
    double min_collision_ttc_s = 1e30;

    double first_elapsed_s = 0.0;
    double last_elapsed_s = 0.0;
    bool has_first_sample = false;
  };

  void UpdatePose(const ResearchLogFrame& frame, double dt_s);
  void UpdateSummary(const ResearchLogFrame& frame, double elapsed_s);
  void WriteHeader();
  void WriteSummaryFile();
  std::string ResolveOutputPath() const;

  static bool ParseEnvBool(const char* value, bool default_value);
  static double Clamp(double x, double lo, double hi);
  static double DegToRad(double deg);
  static double WrapPi(double rad);
  static double FiniteOrNaN(double value);

  ResearchLogOptions options_;
  bool running_ = false;
  std::string output_path_;

  std::ofstream out_;
  uint64_t flush_counter_ = 0;

  std::chrono::steady_clock::time_point start_steady_;

  double route_x_m_ = 0.0;
  double route_y_m_ = 0.0;
  double route_heading_rad_ = 0.0;
  double route_distance_m_ = 0.0;
  int route_source_ = 0;

  Summary summary_;
};
