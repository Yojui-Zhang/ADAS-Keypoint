#include "system_config.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

#include <opencv2/core.hpp>

namespace {

template <typename T>
void ReadIfPresent(const cv::FileNode& node, const char* key, T& out_value) {
    if (node.empty()) return;
    const cv::FileNode n = node[key];
    if (n.empty()) return;
    n >> out_value;
}

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool ParseBoolString(const std::string& s, bool& out_value) {
    const std::string v = ToLower(s);
    if (v == "1" || v == "true" || v == "yes" || v == "on") {
        out_value = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "off") {
        out_value = false;
        return true;
    }
    return false;
}

void ReadBoolIfPresent(const cv::FileNode& node, const char* key, bool& out_value) {
    if (node.empty()) return;
    const cv::FileNode n = node[key];
    if (n.empty()) return;

    if (n.isString()) {
        std::string s;
        n >> s;
        bool b = out_value;
        if (ParseBoolString(s, b)) {
            out_value = b;
        }
        return;
    }

    int v = out_value ? 1 : 0;
    n >> v;
    out_value = (v == 0) ? false : true;
}

void ReadFilterTypeIfPresent(const cv::FileNode& node, const char* key, int& out_value) {
    if (node.empty()) return;
    const cv::FileNode n = node[key];
    if (n.empty()) return;

    if (n.isString()) {
        std::string s;
        n >> s;
        s = ToLower(s);
        if (s == "ema") out_value = 0;
        else if (s == "kf") out_value = 1;
        else if (s == "auto") out_value = -1;
        return;
    }

    n >> out_value;
}

void ReadAppConfig(const cv::FileNode& app_node, AppRuntimeConfig& cfg) {
    ReadIfPresent(app_node, "run_mode", cfg.run_mode);
    ReadIfPresent(app_node, "camera_yaml_path", cfg.camera_yaml_path);
    ReadIfPresent(app_node, "icon_path", cfg.icon_path);
    ReadIfPresent(app_node, "fallback_ego_speed_kmh", cfg.fallback_ego_speed_kmh);
    ReadBoolIfPresent(app_node, "enable_collision_actuation", cfg.enable_collision_actuation);
    ReadBoolIfPresent(app_node, "draw_collision_border", cfg.draw_collision_border);
    ReadBoolIfPresent(app_node, "draw_collision_target_box", cfg.draw_collision_target_box);
    ReadBoolIfPresent(app_node, "show_timing_ms", cfg.show_timing_ms);
    ReadIfPresent(app_node, "wait_key_ms", cfg.wait_key_ms);

    ReadBoolIfPresent(app_node, "enable_keypad_evdev", cfg.enable_keypad_evdev);
    ReadIfPresent(app_node, "keypad_device_path", cfg.keypad_device_path);

    ReadBoolIfPresent(app_node, "can_tx_master_enable", cfg.can_tx_master_enable);
    ReadBoolIfPresent(app_node, "can_longitudinal_enable", cfg.can_longitudinal_enable);
    ReadBoolIfPresent(app_node, "can_steering_enable", cfg.can_steering_enable);
    ReadIfPresent(app_node, "longitudinal_controller", cfg.longitudinal_controller);

    ReadBoolIfPresent(app_node, "draw_inference_overlay", cfg.draw_inference_overlay);
    ReadBoolIfPresent(app_node, "draw_acc_overlay", cfg.draw_acc_overlay);
    ReadBoolIfPresent(app_node, "draw_lka_overlay", cfg.draw_lka_overlay);
    ReadBoolIfPresent(app_node, "draw_behavior_overlay", cfg.draw_behavior_overlay);
    ReadBoolIfPresent(app_node, "draw_collision_overlay", cfg.draw_collision_overlay);
    ReadBoolIfPresent(app_node, "draw_status_hud", cfg.draw_status_hud);
}

void ReadInputConfig(const cv::FileNode& input_node, InputViewConfig& cfg) {
    ReadIfPresent(input_node, "video_path", cfg.video_path);
    ReadIfPresent(input_node, "camera_index", cfg.camera_index);
    ReadIfPresent(input_node, "capture_width", cfg.capture_width);
    ReadIfPresent(input_node, "capture_height", cfg.capture_height);
    ReadIfPresent(input_node, "window_name", cfg.window_name);
    ReadBoolIfPresent(input_node, "fullscreen", cfg.fullscreen);
}

void ReadGeometryConfig(const cv::FileNode& geometry_node, GeometryConfig& cfg) {
    ReadBoolIfPresent(geometry_node, "draw_kpt_world", cfg.draw_kpt_world);
    ReadBoolIfPresent(geometry_node, "draw_box_world", cfg.draw_box_world);
    ReadIfPresent(geometry_node, "world_unit_scale", cfg.world_unit_scale);
}

void ReadModelConfig(const cv::FileNode& model_node, ModelRuntimeConfig& cfg) {
    ReadIfPresent(model_node, "classify_model_width", cfg.classify_model_width);
    ReadIfPresent(model_node, "classify_model_height", cfg.classify_model_height);

    const cv::FileNode trt = model_node["tensorrt"];
    ReadIfPresent(trt, "topk", cfg.tensorrt.topk);
    ReadIfPresent(trt, "score_thres", cfg.tensorrt.score_thres);
    ReadIfPresent(trt, "iou_thres", cfg.tensorrt.iou_thres);
    ReadIfPresent(trt, "num_labels", cfg.tensorrt.num_labels);
}

void ReadSortConfig(const cv::FileNode& sort_node, SORTTRACKING::SortTrackingConfig& cfg) {
    ReadIfPresent(sort_node, "max_age", cfg.max_age);
    ReadIfPresent(sort_node, "min_hits", cfg.min_hits);
    ReadIfPresent(sort_node, "iou_threshold", cfg.iou_threshold);
    ReadIfPresent(sort_node, "history_length", cfg.history_length);
}

void ReadSortKeypointConfig(const cv::FileNode& sort_kpt_node, sort_kpt::KeypointFilterConfig& cfg) {
    ReadFilterTypeIfPresent(sort_kpt_node, "filter_type", cfg.filter_type);
    ReadBoolIfPresent(sort_kpt_node, "allow_env_override", cfg.allow_env_override);

    const cv::FileNode ema_node = sort_kpt_node["ema"];
    ReadIfPresent(ema_node, "conf_thr", cfg.ema_params.conf_thr);
    ReadIfPresent(ema_node, "alpha_lo", cfg.ema_params.alpha_lo);
    ReadIfPresent(ema_node, "alpha_hi", cfg.ema_params.alpha_hi);

    const cv::FileNode kf_node = sort_kpt_node["kf"];
    ReadIfPresent(kf_node, "conf_thr", cfg.kf_params.conf_thr);
    ReadIfPresent(kf_node, "process_var", cfg.kf_params.process_var);
    ReadIfPresent(kf_node, "meas_var_base", cfg.kf_params.meas_var_base);
    ReadIfPresent(kf_node, "gate_dist_px", cfg.kf_params.gate_dist_px);
    ReadIfPresent(kf_node, "init_pos_var", cfg.kf_params.init_pos_var);
    ReadIfPresent(kf_node, "init_vel_var", cfg.kf_params.init_vel_var);
}

void ReadAccConfig(const cv::FileNode& acc_node, acc::AccConfig& cfg) {
    ReadIfPresent(acc_node, "lateral_limit_m", cfg.lateral_limit_m);
    ReadIfPresent(acc_node, "min_forward_m", cfg.min_forward_m);
    ReadIfPresent(acc_node, "max_forward_m", cfg.max_forward_m);
    ReadIfPresent(acc_node, "lead_hysteresis_m", cfg.lead_hysteresis_m);

    ReadIfPresent(acc_node, "cruise_speed_kmh", cfg.cruise_speed_kmh);
    ReadIfPresent(acc_node, "time_gap_s", cfg.time_gap_s);
    ReadIfPresent(acc_node, "standstill_gap_m", cfg.standstill_gap_m);

    ReadIfPresent(acc_node, "max_accel_mps2", cfg.max_accel_mps2);
    ReadIfPresent(acc_node, "comfort_decel_mps2", cfg.comfort_decel_mps2);
    ReadIfPresent(acc_node, "max_decel_mps2", cfg.max_decel_mps2);
    ReadIfPresent(acc_node, "jerk_limit_mps3", cfg.jerk_limit_mps3);

    ReadIfPresent(acc_node, "brake_full_decel_mps2", cfg.brake_full_decel_mps2);
    ReadIfPresent(acc_node, "brake_multiplier", cfg.brake_multiplier);

    ReadIfPresent(acc_node, "default_fps", cfg.default_fps);
    ReadBoolIfPresent(acc_node, "use_external_ego_speed", cfg.use_external_ego_speed);
}

void ReadLkaConfig(const cv::FileNode& lka_node, ControlConfig& cfg) {
    ReadIfPresent(lka_node, "wheel_base_m", cfg.wheel_base_m);
    ReadIfPresent(lka_node, "velocity_mps", cfg.velocity_mps);
    ReadIfPresent(lka_node, "softening", cfg.softening);

    ReadIfPresent(lka_node, "k_straight", cfg.k_straight);
    ReadIfPresent(lka_node, "k_curve", cfg.k_curve);

    ReadIfPresent(lka_node, "x_ref_straight_m", cfg.x_ref_straight_m);
    ReadIfPresent(lka_node, "x_heading_straight_m", cfg.x_heading_straight_m);
    ReadIfPresent(lka_node, "x_ref_curve_m", cfg.x_ref_curve_m);
    ReadIfPresent(lka_node, "x_heading_curve_m", cfg.x_heading_curve_m);

    ReadBoolIfPresent(lka_node, "enable_feedforward", cfg.enable_feedforward);
    ReadIfPresent(lka_node, "ff_gain", cfg.ff_gain);
    ReadIfPresent(lka_node, "x_curvature_m", cfg.x_curvature_m);
    ReadIfPresent(lka_node, "max_ff_deg", cfg.max_ff_deg);

    ReadIfPresent(lka_node, "max_steer_deg", cfg.max_steer_deg);
    ReadIfPresent(lka_node, "max_steer_rate_deg_s", cfg.max_steer_rate_deg_s);
    ReadIfPresent(lka_node, "dt_s", cfg.dt_s);

    ReadBoolIfPresent(lka_node, "use_confidence", cfg.use_confidence);
    ReadIfPresent(lka_node, "conf_threshold", cfg.conf_threshold);
    ReadIfPresent(lka_node, "min_x_m", cfg.min_x_m);
    ReadIfPresent(lka_node, "max_x_m", cfg.max_x_m);
    ReadIfPresent(lka_node, "max_abs_y_m", cfg.max_abs_y_m);

    ReadIfPresent(lka_node, "curvature_samples", cfg.curvature_samples);
    ReadIfPresent(lka_node, "metric_w_mean", cfg.metric_w_mean);
    ReadIfPresent(lka_node, "metric_w_std", cfg.metric_w_std);

    ReadBoolIfPresent(lka_node, "use_sigmoid_probability", cfg.use_sigmoid_probability);
    ReadIfPresent(lka_node, "metric_threshold", cfg.metric_threshold);
    ReadIfPresent(lka_node, "metric_sensitivity", cfg.metric_sensitivity);

    ReadBoolIfPresent(lka_node, "use_hysteresis", cfg.use_hysteresis);
    ReadIfPresent(lka_node, "metric_enter_curve", cfg.metric_enter_curve);
    ReadIfPresent(lka_node, "metric_exit_curve", cfg.metric_exit_curve);

    ReadBoolIfPresent(lka_node, "enable_prob_lowpass", cfg.enable_prob_lowpass);
    ReadIfPresent(lka_node, "prob_alpha", cfg.prob_alpha);

    ReadIfPresent(lka_node, "lane_width_m", cfg.lane_width_m);
    ReadIfPresent(lka_node, "lane_center_offset_m", cfg.lane_center_offset_m);
    ReadIfPresent(lka_node, "visual_limit_m", cfg.visual_limit_m);
}

void ReadStabilityConfig(const cv::FileNode& s_node, stability::StabilityConfig& cfg) {
    ReadIfPresent(s_node, "mass_kg", cfg.mass_kg);
    ReadIfPresent(s_node, "wheelbase_m", cfg.wheelbase_m);
    ReadIfPresent(s_node, "steering_ratio", cfg.steering_ratio);
    ReadIfPresent(s_node, "g", cfg.g);

    ReadIfPresent(s_node, "mu_static", cfg.mu_static);
    ReadIfPresent(s_node, "mu_dynamic", cfg.mu_dynamic);
    ReadIfPresent(s_node, "mu_lowpass_alpha", cfg.mu_lowpass_alpha);

    ReadIfPresent(s_node, "lat_safety", cfg.lat_safety);
    ReadIfPresent(s_node, "total_safety", cfg.total_safety);

    ReadIfPresent(s_node, "lat_accel_comfort_mps2", cfg.lat_accel_comfort_mps2);
    ReadIfPresent(s_node, "long_accel_comfort_mps2", cfg.long_accel_comfort_mps2);
    ReadIfPresent(s_node, "long_decel_comfort_mps2", cfg.long_decel_comfort_mps2);

    ReadIfPresent(s_node, "emergency_decel_cap_mps2", cfg.emergency_decel_cap_mps2);
    ReadIfPresent(s_node, "steer_high_speed_guard_kmh", cfg.steer_high_speed_guard_kmh);
    ReadIfPresent(s_node, "min_speed_for_curvelimit_mps", cfg.min_speed_for_curvelimit_mps);

    ReadIfPresent(s_node, "speed_lowpass_alpha", cfg.speed_lowpass_alpha);
    ReadIfPresent(s_node, "max_speed_drop_mps2", cfg.max_speed_drop_mps2);
    ReadIfPresent(s_node, "max_speed_rise_mps2", cfg.max_speed_rise_mps2);

    ReadIfPresent(s_node, "slip_enter_ratio", cfg.slip_enter_ratio);
    ReadIfPresent(s_node, "slip_exit_ratio", cfg.slip_exit_ratio);

    ReadIfPresent(s_node, "ttc_hard_guard_s", cfg.ttc_hard_guard_s);
    ReadIfPresent(s_node, "w_lat", cfg.w_lat);
    ReadIfPresent(s_node, "w_long", cfg.w_long);

    ReadIfPresent(s_node, "max_jerk_acc_mps3", cfg.max_jerk_acc_mps3);
    ReadIfPresent(s_node, "max_jerk_dec_mps3", cfg.max_jerk_dec_mps3);

    ReadBoolIfPresent(s_node, "use_measured_alat", cfg.use_measured_alat);
    ReadIfPresent(s_node, "alat_lpf_alpha", cfg.alat_lpf_alpha);
    ReadIfPresent(s_node, "alat_meas_timeout_s", cfg.alat_meas_timeout_s);
    ReadIfPresent(s_node, "alat_cmd_guard_ratio", cfg.alat_cmd_guard_ratio);

    ReadBoolIfPresent(s_node, "use_uncertainty_scaling", cfg.use_uncertainty_scaling);
    ReadIfPresent(s_node, "uncert_gain", cfg.uncert_gain);
    ReadIfPresent(s_node, "uncert_sigma_dist_max", cfg.uncert_sigma_dist_max);
    ReadIfPresent(s_node, "ttc_sigma_k", cfg.ttc_sigma_k);
}

void ReadCollisionConfig(const cv::FileNode& c_node, collision::CollisionAssistConfig& cfg) {
    ReadIfPresent(c_node, "roi_y_half_m", cfg.roi_y_half_m);
    ReadIfPresent(c_node, "roi_x_min_m", cfg.roi_x_min_m);
    ReadIfPresent(c_node, "roi_x_max_m", cfg.roi_x_max_m);

    ReadIfPresent(c_node, "danger_forward_m", cfg.danger_forward_m);
    ReadIfPresent(c_node, "corridor_half_width_m", cfg.corridor_half_width_m);
    ReadIfPresent(c_node, "path_sample_step_m", cfg.path_sample_step_m);

    ReadIfPresent(c_node, "horizon_s", cfg.horizon_s);
    ReadIfPresent(c_node, "step_s", cfg.step_s);

    ReadIfPresent(c_node, "ttc_warn_s", cfg.ttc_warn_s);
    ReadIfPresent(c_node, "ttc_brake_s", cfg.ttc_brake_s);
    ReadIfPresent(c_node, "dis_warn_m", cfg.dis_warn_m);
    ReadIfPresent(c_node, "dis_brake_m", cfg.dis_brake_m);

    ReadIfPresent(c_node, "max_extra_brake_0_10", cfg.max_extra_brake_0_10);
    ReadIfPresent(c_node, "max_avoid_steer_deg", cfg.max_avoid_steer_deg);

    ReadBoolIfPresent(c_node, "enable_classify_warning", cfg.enable_classify_warning);
    ReadBoolIfPresent(c_node, "prefer_trackingbox_classify", cfg.prefer_trackingbox_classify);
    ReadIfPresent(c_node, "classify_ttl_frames", cfg.classify_ttl_frames);
    ReadBoolIfPresent(c_node, "enable_bbox_fallback_classify", cfg.enable_bbox_fallback_classify);

    ReadIfPresent(c_node, "classify_car_roi_x_min_px", cfg.classify_car_roi_x_min_px);
    ReadIfPresent(c_node, "classify_car_roi_x_max_px", cfg.classify_car_roi_x_max_px);
    ReadIfPresent(c_node, "classify_car_roi_y_min_px", cfg.classify_car_roi_y_min_px);
    ReadIfPresent(c_node, "classify_center_x_px", cfg.classify_center_x_px);
    ReadIfPresent(c_node, "classify_lr_deadband_px", cfg.classify_lr_deadband_px);

    ReadBoolIfPresent(c_node, "enable_warning_kf", cfg.enable_warning_kf);
    ReadIfPresent(c_node, "warning_kf_q_per_s", cfg.warning_kf_q_per_s);
    ReadIfPresent(c_node, "warning_kf_r", cfg.warning_kf_r);
    ReadIfPresent(c_node, "warning_kf_on_th", cfg.warning_kf_on_th);
    ReadIfPresent(c_node, "warning_kf_off_th", cfg.warning_kf_off_th);

    ReadIfPresent(c_node, "track_warmup_frames", cfg.track_warmup_frames);
    ReadIfPresent(c_node, "attention_half_width_m", cfg.attention_half_width_m);
    ReadIfPresent(c_node, "min_approach_speed_mps", cfg.min_approach_speed_mps);
    ReadIfPresent(c_node, "threat_hold_frames", cfg.threat_hold_frames);
    ReadIfPresent(c_node, "threat_switch_hysteresis_s", cfg.threat_switch_hysteresis_s);

    ReadIfPresent(c_node, "tracker_alpha", cfg.tracker_alpha);
    ReadIfPresent(c_node, "tracker_beta", cfg.tracker_beta);
    ReadIfPresent(c_node, "tracker_dt_min_s", cfg.tracker_dt_min_s);
    ReadIfPresent(c_node, "tracker_dt_max_s", cfg.tracker_dt_max_s);
    ReadIfPresent(c_node, "tracker_stale_frames", cfg.tracker_stale_frames);
    ReadIfPresent(c_node, "tracker_residual_reset_m", cfg.tracker_residual_reset_m);
    ReadIfPresent(c_node, "tracker_vel_max_mps", cfg.tracker_vel_max_mps);
    ReadIfPresent(c_node, "heading_fusion_alpha", cfg.heading_fusion_alpha);
}

void ReadBehaviorConfig(const cv::FileNode& b_node, VehicleBehaviorRuntimeConfig& cfg) {
    ReadBoolIfPresent(b_node, "enable", cfg.enable);
    ReadBoolIfPresent(b_node, "use_custom_layout", cfg.use_custom_layout);

    const cv::FileNode layout_node = b_node["custom_layout"];
    if (layout_node.empty() || layout_node.isSeq() == false) return;
    if (layout_node.size() < cfg.custom_layout.size()) return;

    for (size_t i = 0; i < cfg.custom_layout.size(); ++i) {
        cfg.custom_layout[i] = static_cast<int>(layout_node[static_cast<int>(i)]);
    }
}

void ReadAblationConfig(const cv::FileNode& a_node, AblationRuntimeConfig& cfg) {
    ReadBoolIfPresent(a_node, "enable", cfg.enable);
    ReadIfPresent(a_node, "output_path", cfg.output_path);
    ReadIfPresent(a_node, "output_dir", cfg.output_dir);
    ReadIfPresent(a_node, "flush_every_n", cfg.flush_every_n);
    ReadIfPresent(a_node, "plot_size_px", cfg.plot_size_px);
    ReadIfPresent(a_node, "plot_margin_px", cfg.plot_margin_px);

    ReadBoolIfPresent(a_node, "virtual_road_enable", cfg.virtual_road_enable);
    ReadIfPresent(a_node, "virtual_road_mode", cfg.virtual_road_mode);
    ReadIfPresent(a_node, "virtual_road_csv_path", cfg.virtual_road_csv_path);
    ReadIfPresent(a_node, "virtual_road_length_m", cfg.virtual_road_length_m);
    ReadIfPresent(a_node, "virtual_road_step_m", cfg.virtual_road_step_m);
    ReadIfPresent(a_node, "virtual_road_lane_width_m", cfg.virtual_road_lane_width_m);
    ReadIfPresent(a_node, "virtual_road_arc_radius_m", cfg.virtual_road_arc_radius_m);
    ReadIfPresent(a_node, "virtual_road_s_amplitude_m", cfg.virtual_road_s_amplitude_m);
    ReadIfPresent(a_node, "virtual_road_s_wavelength_m", cfg.virtual_road_s_wavelength_m);

    int sim_frame_count = static_cast<int>(cfg.virtual_sim_frame_count);
    ReadIfPresent(a_node, "virtual_sim_frame_count", sim_frame_count);
    if (sim_frame_count > 0) {
        cfg.virtual_sim_frame_count = static_cast<uint64_t>(sim_frame_count);
    }
    ReadIfPresent(a_node, "virtual_sim_dt_s", cfg.virtual_sim_dt_s);
    ReadIfPresent(a_node, "virtual_sim_speed_kmh", cfg.virtual_sim_speed_kmh);
    ReadIfPresent(a_node, "virtual_sim_max_steer_deg", cfg.virtual_sim_max_steer_deg);
    ReadIfPresent(a_node, "virtual_sim_vc_k_cte", cfg.virtual_sim_vc_k_cte);
    ReadIfPresent(a_node, "virtual_sim_vc_k_heading", cfg.virtual_sim_vc_k_heading);
    ReadIfPresent(a_node, "virtual_sim_raw_k_cte", cfg.virtual_sim_raw_k_cte);
    ReadIfPresent(a_node, "virtual_sim_raw_k_heading", cfg.virtual_sim_raw_k_heading);
    ReadIfPresent(a_node, "virtual_sim_raw_steer_bias_deg", cfg.virtual_sim_raw_steer_bias_deg);
    ReadIfPresent(a_node, "virtual_sim_raw_steer_osc_amp_deg", cfg.virtual_sim_raw_steer_osc_amp_deg);
    ReadIfPresent(a_node, "virtual_sim_raw_steer_osc_period_s", cfg.virtual_sim_raw_steer_osc_period_s);
}

}  // namespace

bool LoadSystemConfig(const std::string& path, AdasSystemConfig& out_config, std::string* out_error) {
    out_config = AdasSystemConfig{};

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (fs.isOpened() == false) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "Failed to open system config: " << path;
            *out_error = oss.str();
        }
        return false;
    }

    ReadAppConfig(fs["app"], out_config.app);
    ReadInputConfig(fs["input"], out_config.input);
    ReadGeometryConfig(fs["geometry"], out_config.geometry);
    ReadModelConfig(fs["model"], out_config.model);
    ReadSortConfig(fs["sort"], out_config.sort);
    ReadSortKeypointConfig(fs["sort_keypoint"], out_config.sort_keypoint);
    ReadAccConfig(fs["acc"], out_config.acc);
    ReadLkaConfig(fs["lka"], out_config.lka);
    ReadStabilityConfig(fs["stability"], out_config.stability);
    ReadCollisionConfig(fs["collision"], out_config.collision);
    ReadBehaviorConfig(fs["behavior"], out_config.behavior);
    ReadAblationConfig(fs["ablation"], out_config.ablation);

    return true;
}
