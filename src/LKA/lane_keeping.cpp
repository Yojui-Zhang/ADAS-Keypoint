#include "lane_keeping.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>

#include "lk_centerline.h"
#include "lk_math.h"
#include "lk_visualization.h"

namespace {
std::mutex g_lane_keeping_mtx;
ControlConfig g_lane_keeping_cfg;
ControlState g_lane_keeping_state;

void ApplyLkaSpeedProfile(const LkaSpeedProfile& profile,
                          ControlConfig* cfg) {
    if (cfg == nullptr) {
        return;
    }

    cfg->lateral_controller = profile.lateral_controller;
    cfg->softening = profile.softening;
    cfg->k_straight = profile.k_straight;
    cfg->k_curve = profile.k_curve;
    cfg->mpc_horizon = profile.mpc_horizon;
    cfg->mpc_q_cte = profile.mpc_q_cte;
    cfg->mpc_q_heading = profile.mpc_q_heading;
    cfg->mpc_q_steer = profile.mpc_q_steer;
    cfg->mpc_r_steer_rate = profile.mpc_r_steer_rate;
    cfg->x_ref_straight_m = profile.x_ref_straight_m;
    cfg->x_heading_straight_m = profile.x_heading_straight_m;
    cfg->x_ref_curve_m = profile.x_ref_curve_m;
    cfg->x_heading_curve_m = profile.x_heading_curve_m;
    cfg->enable_feedforward = profile.enable_feedforward;
    cfg->ff_gain = profile.ff_gain;
    cfg->x_curvature_m = profile.x_curvature_m;
    cfg->max_ff_deg = profile.max_ff_deg;
    cfg->max_steer_deg = profile.max_steer_deg;
    cfg->max_steer_rate_deg_s = profile.max_steer_rate_deg_s;
    cfg->dt_s = profile.dt_s;
}

const LkaSpeedProfile* SelectLkaSpeedProfile(const ControlConfig& cfg,
                                             float speed_kmh) {
    if (cfg.speed_profiles_enable == false) {
        return nullptr;
    }

    const LkaSpeedProfile* first_enabled = nullptr;
    const LkaSpeedProfile* last_enabled = nullptr;
    for (const auto& profile : cfg.speed_profiles) {
        if (profile.enabled == false) {
            continue;
        }

        if (first_enabled == nullptr) {
            first_enabled = &profile;
        }
        last_enabled = &profile;

        const float min_speed = std::min(profile.min_speed_kmh, profile.max_speed_kmh);
        const float max_speed = std::max(profile.min_speed_kmh, profile.max_speed_kmh);
        if (speed_kmh >= min_speed && speed_kmh < max_speed) {
            return &profile;
        }
    }

    if (first_enabled != nullptr && speed_kmh < first_enabled->min_speed_kmh) {
        return first_enabled;
    }
    return last_enabled;
}

float ResolveRuntimeDtS(float frame_dt_s, float fallback_dt_s) {
    if (std::isfinite(frame_dt_s) && frame_dt_s > 1e-6f) {
        return std::clamp(frame_dt_s, 0.001f, 1.0f);
    }
    if (std::isfinite(fallback_dt_s) && fallback_dt_s > 1e-6f) {
        return std::clamp(fallback_dt_s, 0.001f, 1.0f);
    }
    return 0.02f;
}

float DynamicPreviewDistanceM(float velocity_mps, float dt_s, float weight) {
    if (!std::isfinite(velocity_mps) ||
        !std::isfinite(dt_s) ||
        !std::isfinite(weight)) {
        return 0.0f;
    }
    return std::max(0.0f, velocity_mps) *
           std::max(0.0f, dt_s) *
           std::max(0.0f, weight);
}

void ApplyRuntimeTimingAndPreview(float frame_dt_s,
                                  const LkaSpeedProfile* profile,
                                  ControlConfig* cfg,
                                  float* out_dynamic_preview_m) {
    if (cfg == nullptr) {
        return;
    }

    cfg->dt_s = ResolveRuntimeDtS(frame_dt_s, cfg->dt_s);
    const float dynamic_preview_m =
        DynamicPreviewDistanceM(cfg->velocity_mps,
                                cfg->dt_s,
                                cfg->dynamic_preview_distance_weight);
    if (out_dynamic_preview_m != nullptr) {
        *out_dynamic_preview_m = dynamic_preview_m;
    }

    if (profile == nullptr || profile->has_x_ref_straight_m == false) {
        cfg->x_ref_straight_m = dynamic_preview_m;
    }
    if (profile == nullptr || profile->has_x_heading_straight_m == false) {
        cfg->x_heading_straight_m = dynamic_preview_m;
    }
    if (profile == nullptr || profile->has_x_ref_curve_m == false) {
        cfg->x_ref_curve_m = dynamic_preview_m;
    }
    if (profile == nullptr || profile->has_x_heading_curve_m == false) {
        cfg->x_heading_curve_m = dynamic_preview_m;
    }
}

ControlConfig MakeEffectiveControlConfigForSpeed(const ControlConfig& base_cfg,
                                                 float velocity_mps,
                                                 float frame_dt_s,
                                                 std::string* out_debug) {
    ControlConfig effective_cfg = base_cfg;
    effective_cfg.velocity_mps = velocity_mps;

    const float speed_kmh = std::max(0.0f, velocity_mps * 3.6f);
    const LkaSpeedProfile* profile =
        SelectLkaSpeedProfile(base_cfg, speed_kmh);
    if (profile != nullptr) {
        ApplyLkaSpeedProfile(*profile, &effective_cfg);
        float dynamic_preview_m = 0.0f;
        ApplyRuntimeTimingAndPreview(frame_dt_s, profile, &effective_cfg, &dynamic_preview_m);
        if (out_debug != nullptr) {
            std::ostringstream oss;
            oss << "speed_profile=" << profile->min_speed_kmh
                << "-" << profile->max_speed_kmh << "km/h"
                << " speed_kmh=" << speed_kmh
                << " dt_s=" << effective_cfg.dt_s
                << " dynamic_preview_weight=" << effective_cfg.dynamic_preview_distance_weight
                << " dynamic_preview_m=" << dynamic_preview_m
                << " controller=" << effective_cfg.lateral_controller
                << " softening=" << effective_cfg.softening
                << " k_straight=" << effective_cfg.k_straight
                << " k_curve=" << effective_cfg.k_curve
                << " mpc_horizon=" << effective_cfg.mpc_horizon
                << " q_cte=" << effective_cfg.mpc_q_cte
                << " q_heading=" << effective_cfg.mpc_q_heading
                << " q_steer=" << effective_cfg.mpc_q_steer
                << " r_steer_rate=" << effective_cfg.mpc_r_steer_rate
                << " x_ref_s=" << effective_cfg.x_ref_straight_m
                << " x_heading_s=" << effective_cfg.x_heading_straight_m
                << " x_ref_c=" << effective_cfg.x_ref_curve_m
                << " x_heading_c=" << effective_cfg.x_heading_curve_m;
            *out_debug = oss.str();
        }
    } else {
        float dynamic_preview_m = 0.0f;
        ApplyRuntimeTimingAndPreview(frame_dt_s, nullptr, &effective_cfg, &dynamic_preview_m);
        if (out_debug != nullptr) {
            std::ostringstream oss;
            oss << "speed_profile=base speed_kmh=" << speed_kmh
                << " dt_s=" << effective_cfg.dt_s
                << " dynamic_preview_weight=" << effective_cfg.dynamic_preview_distance_weight
                << " dynamic_preview_m=" << dynamic_preview_m
                << " x_ref_s=" << effective_cfg.x_ref_straight_m
                << " x_heading_s=" << effective_cfg.x_heading_straight_m
                << " x_ref_c=" << effective_cfg.x_ref_curve_m
                << " x_heading_c=" << effective_cfg.x_heading_curve_m;
            *out_debug = oss.str();
        }
    }

    return effective_cfg;
}
}  // namespace

void lane_keeping_set_control_config(const ControlConfig& cfg) {
    std::lock_guard<std::mutex> lk(g_lane_keeping_mtx);
    g_lane_keeping_cfg = cfg;
}

ControlConfig lane_keeping_get_control_config() {
    std::lock_guard<std::mutex> lk(g_lane_keeping_mtx);
    return g_lane_keeping_cfg;
}

void lane_keeping_reset_state() {
    std::lock_guard<std::mutex> lk(g_lane_keeping_mtx);
    g_lane_keeping_state = ControlState{};
}

LkaReferenceSnapshot lane_keeping_get_last_reference_snapshot() {
    std::lock_guard<std::mutex> lk(g_lane_keeping_mtx);
    return g_lane_keeping_state.reference_snapshot;
}

float lane_steering_step(const std::vector<TrackingBox>& world_result,
                         float velocity_mps,
                         std::string* out_debug,
                         cv::Mat input_img,
                         cv::Mat output_img,
                         const CameraModel* cam,
                         float frame_dt_s)
{
    std::lock_guard<std::mutex> lk(g_lane_keeping_mtx);

    (void)input_img; // currently unused; preserved for API compatibility

    g_lane_keeping_cfg.velocity_mps = velocity_mps;
    std::string profile_debug;
    const ControlConfig config =
        MakeEffectiveControlConfigForSpeed(g_lane_keeping_cfg,
                                           velocity_mps,
                                           frame_dt_s,
                                           &profile_debug);
    ControlState& state = g_lane_keeping_state;

    // A) WorldResult -> centerline TrackingBox
    TrackingBox center_lane;
    std::string debug_a;
    const bool has_lane = lane_keeping::internal::BuildCenterlineFromWorldResult(
        world_result, config, center_lane, debug_a);

    // Optional visualization
    if (has_lane && !output_img.empty() && cam != nullptr) {
        lane_keeping::internal::DrawFittedLeftRightLanesOnImage(world_result, output_img, *cam, config);
        if (has_lane) {
            lane_keeping::internal::DrawCenterlineOnImage(center_lane, output_img, *cam);
        }
    }

    // B/C/D) centerline -> steering
    if (has_lane) {
        const float steer_deg = calculate_lane_steering(center_lane, config, &state);
        if (out_debug) {
            *out_debug = debug_a + " | " + profile_debug + " | " + state.debug;
        }
        return steer_deg;
    }

    // No lane: steer to zero with rate limiting
    const double target_rad = 0.0;
    const double cmd_rad = lane_keeping::internal::RateLimitRad(
        target_rad,
        static_cast<double>(state.last_steer_rad),
        static_cast<double>(config.max_steer_rate_deg_s),
        static_cast<double>(config.dt_s));

    state.last_steer_rad = static_cast<float>(cmd_rad);
    state.last_steer_deg = static_cast<float>(lane_keeping::internal::Rad2Deg(cmd_rad));
    state.p_curve = 0.0f;
    state.mode_curve = false;
    state.reference_snapshot = LkaReferenceSnapshot{};

    if (out_debug) {
        std::ostringstream oss;
        oss << debug_a << " | " << profile_debug
            << " | no lane -> steer to 0 with rate_limit, steer_deg=" << state.last_steer_deg;
        *out_debug = oss.str();
    }

    return state.last_steer_deg;
}
