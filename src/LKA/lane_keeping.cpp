#include "lane_keeping.h"

#include <sstream>

#include "lk_centerline.h"
#include "lk_math.h"
#include "lk_visualization.h"

float lane_steering_step(const std::vector<TrackingBox>& world_result,
                         float velocity_mps,
                         std::string* out_debug,
                         cv::Mat input_img,
                         cv::Mat output_img,
                         const CameraModel* cam)
{
    (void)input_img; // currently unused; preserved for API compatibility

    // Static config init once (kept identical to original values)
    static ControlConfig config = [](){
        ControlConfig c;

        c.wheel_base_m = 0.30f;
        c.dt_s = 0.02f;

        c.k_straight = 0.7f;
        c.k_curve    = 3.0f;

        // straight: look further; curve: look closer
        c.x_heading_straight_m = 1.5f;
        c.x_heading_curve_m    = 0.8f;

        // probability smoothing
        c.enable_prob_lowpass = true;
        c.prob_alpha = 0.85f;

        // If you want hysteresis mode, switch here:
        // c.use_hysteresis = true;
        // c.use_sigmoid_probability = false;

        return c;
    }();

    static ControlState state;

    // Per-frame speed update
    config.velocity_mps = velocity_mps;

    // A) WorldResult -> centerline TrackingBox
    TrackingBox center_lane;
    std::string debug_a;
    const bool has_lane = lane_keeping::internal::BuildCenterlineFromWorldResult(
        world_result, config, center_lane, debug_a);

    // Optional visualization
    if (has_lane && !output_img.empty() && cam != nullptr) {
        lane_keeping::internal::DrawCenterlineOnImage(center_lane, output_img, *cam);
    }

    // B/C/D) centerline -> steering
    if (has_lane) {
        const float steer_deg = calculate_lane_steering(center_lane, config, &state);
        if (out_debug) {
            *out_debug = debug_a + " | " + state.debug;
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

    if (out_debug) {
        std::ostringstream oss;
        oss << debug_a << " | no lane -> steer to 0 with rate_limit, steer_deg=" << state.last_steer_deg;
        *out_debug = oss.str();
    }

    return state.last_steer_deg;
}
