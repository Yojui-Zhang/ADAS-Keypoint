#include "lane_keeping.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace {

constexpr double kPI = 3.14159265358979323846;

inline double rad2deg(double r) { return r * 180.0 / kPI; }
inline double deg2rad(double d) { return d * kPI / 180.0; }

inline double clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

inline double safe_atan2(double y, double x) {
    return std::atan2(y, x);
}

// 角度差正規化到 [-pi, pi]
inline double wrapPi(double a) {
    while (a >  kPI) a -= 2.0 * kPI;
    while (a < -kPI) a += 2.0 * kPI;
    return a;
}

// 依 x 由小到大排序（近到遠）
void sort_by_x(std::vector<cv::Point2f>& pts) {
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
}

// 擬合 y(x) = a2 x^2 + a1 x + a0 （最小平方法）
// 回傳 coeff = {a2, a1, a0}
bool fit_quadratic(const std::vector<cv::Point2f>& pts,
                   cv::Vec3d& coeff,
                   std::string& dbg)
{
    if (pts.size() < 3) {
        dbg = "fit_quadratic: need >= 3 pts.";
        return false;
    }

    const int N = static_cast<int>(pts.size());
    cv::Mat A(N, 3, CV_64F);
    cv::Mat b(N, 1, CV_64F);

    for (int i = 0; i < N; ++i) {
        const double x = pts[i].x;
        const double y = pts[i].y;
        A.at<double>(i, 0) = x * x;
        A.at<double>(i, 1) = x;
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = y;
    }

    cv::Mat sol;
    const bool ok = cv::solve(A, b, sol, cv::DECOMP_SVD);
    if (!ok || sol.rows != 3) {
        dbg = "fit_quadratic: cv::solve failed.";
        return false;
    }

    coeff[0] = sol.at<double>(0, 0);
    coeff[1] = sol.at<double>(1, 0);
    coeff[2] = sol.at<double>(2, 0);

    std::ostringstream oss;
    oss << "poly a2=" << coeff[0] << " a1=" << coeff[1] << " a0=" << coeff[2];
    dbg = oss.str();
    return true;
}

inline double poly_y(const cv::Vec3d& c, double x) {
    return c[0]*x*x + c[1]*x + c[2];
}

inline double poly_dy_dx(const cv::Vec3d& c, double x) {
    return 2.0*c[0]*x + c[1];
}

inline double poly_d2y_dx2(const cv::Vec3d& c) {
    return 2.0*c[0];
}

// 曲率 κ = y'' / (1 + y'^2)^(3/2)（保留符號）
double curvature_kappa(const cv::Vec3d& c, double x) {
    const double yp  = poly_dy_dx(c, x);
    const double ypp = poly_d2y_dx2(c);
    const double denom = std::pow(1.0 + yp*yp, 1.5);
    if (denom < 1e-9) return 0.0;
    return ypp / denom;
}

// Stanley（回授項）
// delta_fb = heading_err + atan2(k * cte, v + softening)
double stanley_feedback(double cte_m, double heading_err_rad,
                        double v_mps, double k, double softening)
{
    const double v = std::max(0.0, v_mps);
    const double denom = v + std::max(1e-3, softening);
    const double cte_term = std::atan2(k * cte_m, denom);
    return heading_err_rad + cte_term;
}

// 機率計算：由 metric -> p_curve_raw
double metric_to_probability(double metric, const ControlConfig& cfg, bool prev_mode_curve, bool& out_mode_curve)
{
    // hysteresis：先決定離散模式，再輸出 0/1（後續再低通）
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

    // sigmoid：連續機率
    out_mode_curve = prev_mode_curve; // 不使用 hysteresis 時，這個不重要
    const double s = std::max(1e-6f, cfg.metric_sensitivity);
    const double th = cfg.metric_threshold;
    const double p = 1.0 / (1.0 + std::exp(-s * (metric - th)));
    return clamp(p, 0.0, 1.0);
}

// 轉角 rate limit（rad/s）
double rate_limit(double target_rad, double last_rad, double max_rate_deg_s, double dt_s)
{
    if (dt_s <= 1e-6) return target_rad;
    const double max_rate_rad_s = deg2rad(std::max(0.0, (double)max_rate_deg_s));
    const double max_delta = max_rate_rad_s * dt_s;

    const double diff = target_rad - last_rad;
    if (diff >  max_delta) return last_rad + max_delta;
    if (diff < -max_delta) return last_rad - max_delta;
    return target_rad;
}

} // namespace

float calculate_lane_steering(const TrackingBox& input,
                              const ControlConfig& cfg,
                              ControlState* state)
{
    if (!state) return 0.0f;

    // 安全機制：非車道類別或無點 → 回傳上一幀
    if (input.class_id != 0 || input.kpts.empty()) {
        state->debug = "invalid input: class_id!=0 or empty kpts -> hold last steering.";
        return state->last_steer_deg;
    }

    // 1) 取點（本版假設已是 vehicle ground frame: x(m), y(m)）
    std::vector<cv::Point2f> pts;
    pts.reserve(input.kpts.size());

    for (const auto& kp : input.kpts) {
        if (cfg.use_confidence && kp.z < cfg.conf_threshold) continue;

        const float x = kp.x;
        const float y = kp.y;

        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        if (x < cfg.min_x_m || x > cfg.max_x_m) continue;
        if (std::fabs(y) > cfg.max_abs_y_m) continue;

        pts.emplace_back(x, y);
    }

    if (pts.size() < 3) {
        state->debug = "valid pts < 3 after filtering -> hold last steering.";
        return state->last_steer_deg;
    }

    // 2) 排序（近->遠）
    sort_by_x(pts);

    // 3) 二次擬合（B：用多點擬合，穩健算 heading/curvature）
    cv::Vec3d poly;
    std::string fit_dbg;
    if (!fit_quadratic(pts, poly, fit_dbg)) {
        state->debug = "fit failed: " + fit_dbg + " -> hold last steering.";
        return state->last_steer_deg;
    }

    // 4) 估曲率（在多個 x 位置取樣，計算 mean/std）
    const int M = std::max(3, cfg.curvature_samples);

    const double x_min = pts.front().x;
    const double x_max = pts.back().x;
    const double span  = std::max(1e-3, x_max - x_min);

    std::vector<double> kappas;
    kappas.reserve(M);

    for (int i = 0; i < M; ++i) {
        const double t = (M == 1) ? 0.0 : (double)i / (double)(M - 1);
        double xq = x_min + t * span;

        // 也可以固定用 cfg.x_curvature_m（若你希望統一評估距離）
        // 這裡用路徑區間均勻取樣會更穩健
        const double kappa = curvature_kappa(poly, xq);
        kappas.push_back(kappa);
    }

    const double mean_kappa = std::accumulate(kappas.begin(), kappas.end(), 0.0) / (double)kappas.size();

    double var = 0.0;
    for (double k : kappas) {
        const double d = k - mean_kappa;
        var += d * d;
    }
    var /= std::max(1.0, (double)kappas.size());
    const double std_kappa = std::sqrt(var);

    // metric = w1*|mean| + w2*std
    const double metric = (double)cfg.metric_w_mean * std::fabs(mean_kappa)
                        + (double)cfg.metric_w_std  * std_kappa;

    // 5) (C) 由 metric -> p_curve_raw，再做低通
    bool mode_curve_new = state->mode_curve;
    const double p_raw = metric_to_probability(metric, cfg, state->mode_curve, mode_curve_new);

    double p_curve = p_raw;
    if (cfg.enable_prob_lowpass) {
        const double a = clamp(cfg.prob_alpha, 0.0f, 0.999f);
        p_curve = a * state->p_curve + (1.0 - a) * p_raw;
    }
    p_curve = clamp(p_curve, 0.0, 1.0);

    state->p_curve = (float)p_curve;
    state->mode_curve = mode_curve_new;

    // 6) 計算兩模型的 cte / heading（用擬合函數評估）
    auto compute_cte_heading = [&](double x_ref, double x_heading, double& out_cte, double& out_heading_err) {
        const double x_r = clamp(x_ref, x_min, x_max);
        const double x_h = clamp(x_heading, x_min, x_max);

        out_cte = poly_y(poly, x_r);                 // 車在 y=0，路徑 y=cte（左正）
        const double slope = poly_dy_dx(poly, x_h);
        const double psi_path = safe_atan2(slope, 1.0); // atan(dy/dx)
        out_heading_err = wrapPi(psi_path);          // 車頭朝 x 軸 → heading_err = psi_path
    };

    double cte_s=0, head_s=0;
    double cte_c=0, head_c=0;

    compute_cte_heading(cfg.x_ref_straight_m, cfg.x_heading_straight_m, cte_s, head_s);
    compute_cte_heading(cfg.x_ref_curve_m,    cfg.x_heading_curve_m,    cte_c, head_c);

    // 7) 計算兩模型 Stanley 回授（B）
    const double v = std::max(0.0, (double)cfg.velocity_mps);
    const double delta_fb_straight = stanley_feedback(cte_s, head_s, v, cfg.k_straight, cfg.softening);
    const double delta_fb_curve    = stanley_feedback(cte_c, head_c, v, cfg.k_curve,    cfg.softening);

    // 8) (D) 曲率前饋（建議用固定距離 x_curvature_m 或者 mean_kappa）
    double delta_ff = 0.0;
    if (cfg.enable_feedforward) {
        // 這裡使用 mean_kappa 做前饋（穩健；也可改用 curvature_kappa(poly, cfg.x_curvature_m)）
        const double kappa_ff = mean_kappa;
        delta_ff = std::atan((double)cfg.wheel_base_m * kappa_ff) * (double)cfg.ff_gain;

        // 限制前饋避免暴衝
        const double max_ff_rad = deg2rad(cfg.max_ff_deg);
        delta_ff = clamp(delta_ff, -max_ff_rad, +max_ff_rad);
    }

    // 9) 融合：先融合回授，再加前饋（或你也可前饋也做權重）
    const double delta_fb = (1.0 - p_curve) * delta_fb_straight + p_curve * delta_fb_curve;
    double delta_cmd = delta_ff + delta_fb;

    // 10) 總轉角限制
    const double max_steer_rad = deg2rad(cfg.max_steer_deg);
    delta_cmd = clamp(delta_cmd, -max_steer_rad, +max_steer_rad);

    // 11) (D) 轉角速率限制（抑制抖動/致動器限制）
    delta_cmd = rate_limit(delta_cmd, state->last_steer_rad, cfg.max_steer_rate_deg_s, cfg.dt_s);

    // 更新狀態
    state->last_steer_rad = (float)delta_cmd;
    state->last_steer_deg = (float)rad2deg(delta_cmd);

    // debug 資訊
    {
        std::ostringstream oss;
        oss << "ok; " << fit_dbg
            << " | metric=" << metric
            << " p_raw=" << p_raw
            << " p_curve=" << p_curve
            << " mean_kappa=" << mean_kappa
            << " std_kappa=" << std_kappa
            << " | cte_s=" << cte_s << " head_s(rad)=" << head_s
            << " | cte_c=" << cte_c << " head_c(rad)=" << head_c
            << " | delta_ff(deg)=" << rad2deg(delta_ff)
            << " delta_fb(deg)=" << rad2deg(delta_fb)
            << " delta_cmd(deg)=" << state->last_steer_deg;
        state->debug = oss.str();
    }

    return state->last_steer_deg;
}
