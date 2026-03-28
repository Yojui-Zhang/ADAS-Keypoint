#include "lk_stanley_controller.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>

#include "lk_lane_points.h"
#include "lk_math.h"
#include "lk_polyfit.h"

namespace lane_keeping {
namespace internal {

double StanleyFeedback(double cte_m,
                       double heading_err_rad,
                       double v_mps,
                       double k,
                       double softening)
{
    const double v = std::max(0.0, v_mps);
    const double denom = v + std::max(1e-3, softening);
    const double cte_term = std::atan2(k * cte_m, denom);
    return heading_err_rad + cte_term;
}

double MetricToProbability(double metric,
                           const ControlConfig& cfg,
                           bool prev_mode_curve,
                           bool& out_mode_curve)
{
    // hysteresis: decide discrete mode first, then output 0/1
    if (cfg.use_hysteresis) {
        bool mode = prev_mode_curve;
        if (!prev_mode_curve) {
            if (metric >= cfg.metric_enter_curve) mode = true;
        } else {
            if (metric <= cfg.metric_exit_curve) mode = false;
        }
        out_mode_curve = mode;
        return mode ? 1.0 : 0.0;
    }

    // sigmoid: continuous probability
    out_mode_curve = prev_mode_curve; // not meaningful without hysteresis
    const double s = std::max(1e-6, static_cast<double>(cfg.metric_sensitivity));
    const double th = static_cast<double>(cfg.metric_threshold);
    const double p = 1.0 / (1.0 + std::exp(-s * (metric - th)));
    return Clamp(p, 0.0, 1.0);
}

} // namespace internal
} // namespace lane_keeping

// ========================
// Public API implementation
// ========================

namespace {

void ClearReferenceSnapshot(ControlState* state) {
    if (state == nullptr) {
        return;
    }

    state->reference_snapshot = LkaReferenceSnapshot{};
    state->reference_snapshot.p_curve = state->p_curve;
}

}  // namespace

float calculate_lane_steering(const TrackingBox& input,
                              const ControlConfig& cfg,
                              ControlState* state)
{
    using namespace lane_keeping::internal;

    if (!state) return 0.0f;

    // Preserve original safety behavior: invalid class or empty points -> hold last.
    if (input.class_id != 0 || input.kpts.empty()) {
        ClearReferenceSnapshot(state);
        state->debug = "invalid input: class_id!=0 or empty kpts -> hold last steering.";
        return state->last_steer_deg;
    }

    // 1) Extract and filter points (vehicle ground frame, meters)
    std::vector<cv::Point2f> pts;
    std::string extract_dbg;
    const LanePointStatus status = ExtractLanePointsVehicleM(input, cfg, pts, &extract_dbg);

    if (status == LanePointStatus::kNotLane || status == LanePointStatus::kEmpty) {
        ClearReferenceSnapshot(state);
        state->debug = "invalid input: class_id!=0 or empty kpts -> hold last steering.";
        return state->last_steer_deg;
    }
    if (status != LanePointStatus::kOk) {
        ClearReferenceSnapshot(state);
        state->debug = "valid pts < 3 after filtering -> hold last steering.";
        return state->last_steer_deg;
    }

    // 2) Quadratic fit (least squares)
    cv::Vec3d poly;
    std::string fit_dbg;
    if (!FitQuadraticLeastSquares(pts, poly, fit_dbg)) {
        ClearReferenceSnapshot(state);
        state->debug = "fit failed: " + fit_dbg + " -> hold last steering.";
        return state->last_steer_deg;
    }

    // 3) Curvature sampling: mean/std over path interval
    const int M = std::max(3, cfg.curvature_samples);

    const double x_min = pts.front().x;
    const double x_max = pts.back().x;
    const double span = std::max(1e-3, x_max - x_min);

    std::vector<double> kappas;
    kappas.reserve(M);

    for (int i = 0; i < M; ++i) {
        const double t = (M == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(M - 1);
        const double xq = x_min + t * span;
        kappas.push_back(CurvatureKappa(poly, xq));
    }

    const double mean_kappa = std::accumulate(kappas.begin(), kappas.end(), 0.0) /
                              static_cast<double>(kappas.size());

    double var = 0.0;
    for (double k : kappas) {
        const double d = k - mean_kappa;
        var += d * d;
    }
    var /= std::max(1.0, static_cast<double>(kappas.size()));
    const double std_kappa = std::sqrt(var);

    const double metric = static_cast<double>(cfg.metric_w_mean) * std::fabs(mean_kappa) +
                          static_cast<double>(cfg.metric_w_std)  * std_kappa;

    // 4) metric -> p_curve_raw and optional low-pass
    bool mode_curve_new = state->mode_curve;
    const double p_raw = MetricToProbability(metric, cfg, state->mode_curve, mode_curve_new);

    double p_curve = p_raw;
    if (cfg.enable_prob_lowpass) {
        const double a = Clamp(cfg.prob_alpha, 0.0, 0.999);
        p_curve = a * state->p_curve + (1.0 - a) * p_raw;
    }
    p_curve = Clamp(p_curve, 0.0, 1.0);

    state->p_curve = static_cast<float>(p_curve);
    state->mode_curve = mode_curve_new;

    // 5) Compute cte/heading for both models
    auto compute_cte_heading = [&](double x_ref, double x_heading,
                                   double& out_cte, double& out_heading_err)
    {
        const double x_r = Clamp(x_ref, x_min, x_max);
        const double x_h = Clamp(x_heading, x_min, x_max);

        // Shift the target path relative to the lane centerline.
        // Positive offset means track left of the lane center; negative means right.
        out_cte = PolyY(poly, x_r) + static_cast<double>(cfg.lane_center_offset_m);

        const double slope = PolyDyDx(poly, x_h);
        const double psi_path = std::atan2(slope, 1.0); // atan(dy/dx)
        out_heading_err = WrapPi(psi_path); // vehicle heading along +x
    };

    double cte_s = 0.0;
    double head_s = 0.0;
    double cte_c = 0.0;
    double head_c = 0.0;

    compute_cte_heading(cfg.x_ref_straight_m, cfg.x_heading_straight_m, cte_s, head_s);
    compute_cte_heading(cfg.x_ref_curve_m,    cfg.x_heading_curve_m,    cte_c, head_c);

    const double x_ref_s = Clamp(cfg.x_ref_straight_m, x_min, x_max);
    const double x_ref_c = Clamp(cfg.x_ref_curve_m, x_min, x_max);
    const double x_heading_s = Clamp(cfg.x_heading_straight_m, x_min, x_max);
    const double x_heading_c = Clamp(cfg.x_heading_curve_m, x_min, x_max);

    const double target_y_s = PolyY(poly, x_heading_s) + static_cast<double>(cfg.lane_center_offset_m);
    const double target_y_c = PolyY(poly, x_heading_c) + static_cast<double>(cfg.lane_center_offset_m);

    LkaReferenceSnapshot reference_snapshot;
    reference_snapshot.valid = true;
    reference_snapshot.has_lane = true;
    reference_snapshot.p_curve = static_cast<float>(p_curve);
    reference_snapshot.current_point.valid = true;
    reference_snapshot.current_point.x_m =
        static_cast<float>((1.0 - p_curve) * x_ref_s + p_curve * x_ref_c);
    reference_snapshot.current_point.y_m = 0.0f;
    reference_snapshot.target_point.valid = true;
    reference_snapshot.target_point.x_m =
        static_cast<float>((1.0 - p_curve) * x_heading_s + p_curve * x_heading_c);
    reference_snapshot.target_point.y_m =
        static_cast<float>((1.0 - p_curve) * target_y_s + p_curve * target_y_c);
    state->reference_snapshot = reference_snapshot;

    // 6) Stanley feedback for each model
    const double v = std::max(0.0, static_cast<double>(cfg.velocity_mps));

    const double delta_fb_straight = StanleyFeedback(cte_s, head_s, v, cfg.k_straight, cfg.softening);
    const double delta_fb_curve    = StanleyFeedback(cte_c, head_c, v, cfg.k_curve,    cfg.softening);

    // 7) Curvature feedforward
    double delta_ff = 0.0;
    if (cfg.enable_feedforward) {
        const double kappa_ff = mean_kappa;
        delta_ff = std::atan(static_cast<double>(cfg.wheel_base_m) * kappa_ff) *
                   static_cast<double>(cfg.ff_gain);

        const double max_ff_rad = Deg2Rad(cfg.max_ff_deg);
        delta_ff = Clamp(delta_ff, -max_ff_rad, +max_ff_rad);
    }

    // 8) Blend feedback and add feedforward
    const double delta_fb = (1.0 - p_curve) * delta_fb_straight + p_curve * delta_fb_curve;
    double delta_cmd = delta_ff + delta_fb;

    const double max_steer_rad = Deg2Rad(cfg.max_steer_deg);
    delta_cmd = Clamp(delta_cmd, -max_steer_rad, +max_steer_rad);

    delta_cmd = RateLimitRad(delta_cmd,
                             static_cast<double>(state->last_steer_rad),
                             static_cast<double>(cfg.max_steer_rate_deg_s),
                             static_cast<double>(cfg.dt_s));

    state->last_steer_rad = static_cast<float>(delta_cmd);
    state->last_steer_deg = static_cast<float>(Rad2Deg(delta_cmd));

    {
        std::ostringstream oss;
        oss << "ok; " << fit_dbg
            << " | metric=" << metric
            << " p_raw=" << p_raw
            << " p_curve=" << p_curve
            << " mean_kappa=" << mean_kappa
            << " std_kappa=" << std_kappa
            << " | lane_offset_m=" << cfg.lane_center_offset_m
            << " | cte_s=" << cte_s << " head_s(rad)=" << head_s
            << " | cte_c=" << cte_c << " head_c(rad)=" << head_c
            << " | delta_ff(deg)=" << Rad2Deg(delta_ff)
            << " delta_fb(deg)=" << Rad2Deg(delta_fb)
            << " delta_cmd(deg)=" << state->last_steer_deg;
        state->debug = oss.str();
    }

    return state->last_steer_deg;
}
