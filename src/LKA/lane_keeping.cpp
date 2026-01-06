#include "lane_keeping.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

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
// 取出一條 lane box 的有效點（假設已是 vehicle frame: x(m), y(m)）
bool extract_lane_points_vehicle_m(const TrackingBox& box,
                                  const ControlConfig& cfg,
                                  std::vector<cv::Point2f>& out_pts,
                                  std::string& dbg)
{
    out_pts.clear();
    if (box.class_id != 0) {
        // cout << "not lane class_id(0)." << endl;
        return false;
    }
    if (box.kpts.empty()) {
        // cout << "empty kpts." << endl;
        return false;
    }

    for (const auto& kp : box.kpts) {
        if (cfg.use_confidence && kp.z < cfg.conf_threshold) continue;

        const float x = kp.x;
        const float y = kp.y;

        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        if (x < cfg.min_x_m || x > cfg.max_x_m) continue;
        if (std::fabs(y) > cfg.max_abs_y_m) continue;

        out_pts.emplace_back(x, y);
    }

    if (out_pts.size() < 3) {
        // cout << "valid pts < 3 after filtering." << endl;
        return false;
    }

    std::sort(out_pts.begin(), out_pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b){ return a.x < b.x; });

    dbg = "ok pts=" + std::to_string(out_pts.size());

    return true;
}

// 在 pts(x排序) 上，做 y(xq) 線性插值取樣
bool sample_y_linear(const std::vector<cv::Point2f>& pts,
                     float xq,
                     float& yq)
{
    if (pts.size() < 2) return false;
    if (xq < pts.front().x || xq > pts.back().x) return false;

    // 找到第一個 pts[i].x >= xq
    auto it = std::lower_bound(
        pts.begin(), pts.end(), xq,
        [](const cv::Point2f& p, float value){ return p.x < value; });

    if (it == pts.begin()) {
        yq = it->y;
        return true;
    }
    if (it == pts.end()) {
        yq = pts.back().y;
        return true;
    }

    const auto& p2 = *it;
    const auto& p1 = *(it - 1);

    const float dx = p2.x - p1.x;
    if (std::fabs(dx) < 1e-6f) {
        yq = p2.y;
        return true;
    }

    const float t = (xq - p1.x) / dx;
    yq = p1.y + t * (p2.y - p1.y);
    return true;
}

// 以某個 x_eval 估計該 lane 的橫向位置 y_eval，用於判定左右與「離中心最近」
bool estimate_lane_y_at_x(const std::vector<cv::Point2f>& pts,
                          float x_eval,
                          float& y_eval)
{
    // 若 x_eval 超出範圍，改用最近可用點
    if (pts.empty()) return false;

    if (x_eval <= pts.front().x) {
        y_eval = pts.front().y;
        return true;
    }
    if (x_eval >= pts.back().x) {
        y_eval = pts.back().y;
        return true;
    }
    return sample_y_linear(pts, x_eval, y_eval);
}

// -------- 函式 A：讀 WorldResult，選左右最近車道線，合成 centerline --------
// 輸入：WorldResult (多條 TrackingBox)
// 輸出：centerline_box（class_id=0，kpts為中心線 (x,y,conf=1)）
// 回傳：true=成功找到至少一條車道線（左或右），false=完全沒有車道線
bool build_centerline_from_worldresult(const std::vector<TrackingBox>& world_result,
                                       const ControlConfig& cfg,
                                       TrackingBox& centerline_box,
                                       std::string& dbg)
{
    // 你可以把 lane width 放在 cfg 裡；這裡先用常見 3.5m
    const float lane_width_m = 3.5f;
    const float half_lane_m  = lane_width_m * 0.5f;

    // 用於挑「離中心最近」的評估距離
    const float x_eval = std::max(0.5f, std::min(cfg.x_heading_straight_m, 3.0f));

    bool has_left  = false;
    bool has_right = false;

    std::vector<cv::Point2f> best_left_pts;
    std::vector<cv::Point2f> best_right_pts;

    float best_left_abs_y  = 1e9f;
    float best_right_abs_y = 1e9f;

    std::ostringstream oss;

    for (const auto& box : world_result) {
        
        if (box.class_id != 0) continue;
        std::vector<cv::Point2f> pts;
        std::string one_dbg;
        if (!extract_lane_points_vehicle_m(box, cfg, pts, one_dbg)) {
            continue;
        }

        float y_eval = 0.0f;
        if (!estimate_lane_y_at_x(pts, x_eval, y_eval)) {
            continue;
        }

        const float abs_y = std::fabs(y_eval);

        // 左：y>0；右：y<0
        if (y_eval > 0.0f) {
            if (!has_left || abs_y < best_left_abs_y) {
                has_left = true;
                best_left_abs_y = abs_y;
                best_left_pts = std::move(pts);
            }
        } else if (y_eval < 0.0f) {
            if (!has_right || abs_y < best_right_abs_y) {
                has_right = true;
                best_right_abs_y = abs_y;
                best_right_pts = std::move(pts);
            }
        } else {
            // y==0 很少見，視為兩邊都可能；這裡忽略
        }
    }

    if (!has_left && !has_right) {
        // cout << "No lane line found in WorldResult." << endl;
        return false;
    }

    // 建立中心線取樣 x 範圍：取能用的共同範圍，或單邊可用範圍
    float x_min = 0.0f, x_max = 0.0f;

    if (has_left && has_right) {
        x_min = std::max(best_left_pts.front().x,  best_right_pts.front().x);
        x_max = std::min(best_left_pts.back().x,   best_right_pts.back().x);
    } else if (has_left) {
        x_min = best_left_pts.front().x;
        x_max = best_left_pts.back().x;
    } else { // has_right
        x_min = best_right_pts.front().x;
        x_max = best_right_pts.back().x;
    }

    // 確保 x 範圍合理
    x_min = std::max(x_min, cfg.min_x_m);
    x_max = std::min(x_max, cfg.max_x_m);

    if (x_max - x_min < 0.3f) {
        dbg = "A: lane range too small to build centerline.";
        return false;
    }

    // 取樣點數：盡量接近 15（與你 YOLO 輸出一致）
    const int N = 15;
    std::vector<cv::Point3f> center_kpts;
    center_kpts.reserve(N);

    for (int i = 0; i < N; ++i) {
        const float t = (N == 1) ? 0.0f : (float)i / (float)(N - 1);
        const float xq = x_min + t * (x_max - x_min);

        float y_center = 0.0f;

        if (has_left && has_right) {
            float yL = 0.0f, yR = 0.0f;
            const bool okL = sample_y_linear(best_left_pts,  xq, yL);
            const bool okR = sample_y_linear(best_right_pts, xq, yR);
            if (!okL || !okR) continue;

            y_center = 0.5f * (yL + yR); // 中心線 = 左右平均
        } else if (has_left) {
            float yL = 0.0f;
            if (!sample_y_linear(best_left_pts, xq, yL)) continue;
            // 只有左線：中心線在其右側 half_lane
            y_center = yL - half_lane_m;
        } else { // has_right
            float yR = 0.0f;
            if (!sample_y_linear(best_right_pts, xq, yR)) continue;
            // 只有右線：中心線在其左側 half_lane
            y_center = yR + half_lane_m;
        }

        center_kpts.emplace_back(xq, y_center, 1.0f);
    }

    if (center_kpts.size() < 3) {
        dbg = "A: centerline kpts < 3 after sampling.";
        return false;
    }

    centerline_box = TrackingBox{};
    centerline_box.class_id = 0;
    centerline_box.kpts = std::move(center_kpts);

    oss << "A: has_left=" << has_left << " has_right=" << has_right
        << " | x_range=[" << x_min << "," << x_max << "]"
        << " | center_pts=" << centerline_box.kpts.size()
        << " | eval_x=" << x_eval
        << " | best_left_abs_y=" << (has_left ? best_left_abs_y : -1)
        << " | best_right_abs_y=" << (has_right ? best_right_abs_y : -1);

    dbg = oss.str();
    return true;
}

float lane_steering_step(const std::vector<TrackingBox>& world_result,
                         float velocity_mps,
                         std::string* out_debug,
                         cv::Mat input_img,
                         cv::Mat output_img,
                         const CameraModel* cam)
{
    // 固定參數只初始化一次
    static ControlConfig cfg = [](){
        ControlConfig c;

        c.wheel_base_m = 0.30f;
        c.dt_s = 0.02f;

        c.k_straight = 0.7f;
        c.k_curve    = 3.0f;

        // 直線看遠、彎道看近
        c.x_heading_straight_m = 1.5f;
        c.x_heading_curve_m    = 0.8f;

        // 機率平滑
        c.enable_prob_lowpass = true;
        c.prob_alpha = 0.85f;

        // 若你要改用 hysteresis，在這裡切換
        // c.use_hysteresis = true;
        // c.use_sigmoid_probability = false;

        return c;
    }();

    static ControlState st;

    // 每帧更新車速
    cfg.velocity_mps = velocity_mps;

    // 先用函式A把 WorldResult 轉成單一 centerline lane_box
    TrackingBox center_lane;
    std::string dbgA;

    const bool has_lane = build_centerline_from_worldresult(world_result, cfg, center_lane, dbgA);

    float steer_deg = 0.0f;

    // =========================================================
    if (has_lane && !output_img.empty() && cam != nullptr) {
        std::vector<cv::Point> draw_pts;
        
        for (const auto& kp : center_lane.kpts) {
            // center_lane.kpts 格式: Unit=Meters, X=Forward, Y=Left
            
            // 1. 座標逆轉換 (Meter -> CM, Veh Frame -> Raw World Frame)
            // 根據你的 GeometryFunction:
            // x_forward_m = p.y_raw_cm / 100.0  =>  p.y_raw_cm = x_forward_m * 100.0
            // y_left_m = -p.x_raw_cm / 100.0    =>  p.x_raw_cm = -y_left_m * 100.0
            
            float raw_x_cm = -kp.y * 100.0f; 
            float raw_y_cm = kp.x * 100.0f;
            float raw_z_cm = 0.0f; // 假設貼地

            // 2. 投影回像素座標
            cv::Point2f uv = cam->project3dToPixel(cv::Point3f(raw_x_cm, raw_y_cm, raw_z_cm));
            
            // 過濾掉畫面外的點 (可選)
            if (uv.x >= 0 && uv.x < output_img.cols && uv.y >= 0 && uv.y < output_img.rows) {
                draw_pts.push_back(uv);
            }
        }

        // 3. 畫出紅色折線 (Thickness=3)
        if (draw_pts.size() >= 2) {
            cv::polylines(output_img, draw_pts, false, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }
        
        // 4. (選用) 畫出 "目標點" (Heading Reference Point)
        // 找出離車子前方 cfg.x_heading_straight_m 最近的點並標示
        // 這可以幫你視覺化車子正在「看哪裡」
    }
    // =========================================================

    if (has_lane) {
        steer_deg = calculate_lane_steering(center_lane, cfg, &st);
        if (out_debug) {
            *out_debug = dbgA + " | " + st.debug;
        }
        return steer_deg;
    }


    // 都沒有車道線：輸出直走（但用 rate limit 平滑回正）
    const double target_rad = 0.0;
    double cmd_rad = rate_limit(target_rad, st.last_steer_rad, cfg.max_steer_rate_deg_s, cfg.dt_s);

    st.last_steer_rad = (float)cmd_rad;
    st.last_steer_deg = (float)rad2deg(cmd_rad);
    st.p_curve = 0.0f;
    st.mode_curve = false;

    if (out_debug) {
        std::ostringstream oss;
        oss << dbgA << " | no lane -> steer to 0 with rate_limit, steer_deg=" << st.last_steer_deg;
        *out_debug = oss.str();
    }
    return st.last_steer_deg;
}
