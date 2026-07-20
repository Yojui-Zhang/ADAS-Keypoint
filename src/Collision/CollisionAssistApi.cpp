#include "CollisionAssistApi.h"
#include "CollisionRiskModel.h"

#include <cmath>
#include <sstream>
#include <unordered_map>
#include <algorithm>

namespace collision {


namespace {

// C++11 SFINAE: 如果 tb.traffic_class_num 存在就用它，否則回傳 -1
template <typename T>
auto GetTrafficClassNumImpl(const T& tb, int) -> decltype(tb.traffic_class_num, int()) {
  return static_cast<int>(tb.traffic_class_num);
}
template <typename T>
int GetTrafficClassNumImpl(const T&, ...) {
  return -1;
}

// 如果你實際成員叫 traffic_class_id，也一併支援（可刪）
template <typename T>
auto GetTrafficClassIdImpl(const T& tb, int) -> decltype(tb.traffic_class_id, int()) {
  return static_cast<int>(tb.traffic_class_id);
}
template <typename T>
int GetTrafficClassIdImpl(const T&, ...) {
  return -1;
}

static inline int GetTrafficClassSafe(const TrackingBox& tb) {
  int v = GetTrafficClassNumImpl(tb, 0);
  if (v != -1) return v;
  v = GetTrafficClassIdImpl(tb, 0);
  return v;
}

} // namespace

void CollisionAssist::ApplyConfigToTracker()
{
  tracker_.SetGains(cfg_.tracker_alpha, cfg_.tracker_beta);
  tracker_.SetDtClamp(cfg_.tracker_dt_min_s, cfg_.tracker_dt_max_s);
  tracker_.SetStaleFrames(cfg_.tracker_stale_frames);
  tracker_.SetResidualResetMeters(cfg_.tracker_residual_reset_m);
  tracker_.SetVelMaxMps(cfg_.tracker_vel_max_mps);
}


void CollisionAssist::UpdateClassify(int track_id, int frame, int classify_id)
{
  auto& st = classify_by_id_[track_id];
  st.cls = classify_id;
  st.last_frame = frame;
}

std::vector<cv::Point2f> CollisionAssist::SampleEgoPath(float steer_cmd_deg,
                                                        float forward_m,
                                                        float step_m)
{
  const auto scfg = stability::VehicleControl_GetStabilityConfig();

  const double delta_road_rad = stability::Deg2Rad( (double)steer_cmd_deg / std::max(1e-3, scfg.steering_ratio) );
  const double kappa = stability::CurvatureFromSteerRad(delta_road_rad, scfg.wheelbase_m);

  std::vector<cv::Point2f> path;
  const float s_max = std::max(0.0f, forward_m);
  const float ds    = std::max(1e-3f, step_m);

  path.reserve((size_t)(s_max / ds) + 2);

  for (float s = 0.f; s <= s_max + 1e-3f; s += ds) {
    if (std::abs(kappa) < 1e-6) {
      path.emplace_back(s, 0.f);
    } else {
      const double R = 1.0 / kappa;
      const double th = (double)s * kappa;
      const double x = R * std::sin(th);
      const double y = R * (1.0 - std::cos(th));
      path.emplace_back((float)x, (float)y);
    }
  }
  return path;
}

float CollisionAssist::MinDistToPath(const cv::Point2f& p,
                                     const std::vector<cv::Point2f>& path)
{
  float best = std::numeric_limits<float>::infinity();
  for (const auto& q : path) {
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    const float d = std::sqrt(dx*dx + dy*dy);
    if (d < best) best = d;
  }
  return best;
}

CollisionAssistOutput CollisionAssist::Step(const std::vector<TrackingBox>& world_result,
                                            float /*ego_speed_mps*/,
                                            float steer_cmd_deg,
                                            float dt_s,
                                            bool enable_actuation,
                                            float* io_speed_kmh,
                                            float* io_steer_deg,
                                            float* io_brake_0_10)
{
  CollisionAssistOutput out;

  if (!io_speed_kmh || !io_steer_deg || !io_brake_0_10) {
    out.debug = "io_speed_kmh/io_steer_deg/io_brake_0_10 is null";
    return out;
  }

  dt_s = std::clamp(dt_s, 0.005f, 0.2f);

  out.ego_path = SampleEgoPath(steer_cmd_deg, cfg_.danger_forward_m, cfg_.path_sample_step_m);

  // ----------------------------------------------------------------------
  // classify warning (9~12)
  bool classify_warn_raw = false;
  int  classify_warn_track_id = -1;
  int  classify_warn_cls_id   = -1;
  float classify_best_x = std::numeric_limits<float>::infinity();
  cv::Point2f classify_warn_pos{0.f, 0.f};

  auto is_ca_classify = [](int cls) -> bool { return cls >= 9 && cls <= 12; };

  auto get_classify_id = [&](const TrackingBox& tb, const cv::Point2f& /*ground_xy*/) -> int {
    if (!cfg_.enable_classify_warning) return -1;

    // 0) Prefer TrackingBox.traffic_class_num
    if (cfg_.prefer_trackingbox_classify) {
      const int cls = GetTrafficClassSafe(tb);
      return is_ca_classify(cls) ? cls : -1;
    }

    // 1) Optional cache from UpdateClassify()
    auto it = classify_by_id_.find(tb.id);
    if (it != classify_by_id_.end()) {
      if ((tb.frame - it->second.last_frame) <= cfg_.classify_ttl_frames) {
        const int cls = it->second.cls;
        return is_ca_classify(cls) ? cls : -1;
      }
    }

    // 2) bbox fallback (NOT recommended)
    if (!cfg_.enable_bbox_fallback_classify) return -1;
    if (tb.class_id != 1) return -1;
    if (tb.box.x < cfg_.classify_car_roi_x_min_px || tb.box.x > cfg_.classify_car_roi_x_max_px) return -1;
    if (tb.box.y < cfg_.classify_car_roi_y_min_px) return -1;

    const int cx = tb.box.x + tb.box.width / 2;
    if (cx < (cfg_.classify_center_x_px - cfg_.classify_lr_deadband_px)) return 9;
    if (cx > (cfg_.classify_center_x_px + cfg_.classify_lr_deadband_px)) return 11;
    return 10;
  };

  // 1) 更新每台車速度並做軌跡預測
  std::unordered_map<int, bool> seen;
  int frame_now = -1;

  for (const auto& tb : world_result) {
    frame_now = std::max(frame_now, tb.frame);
    if (tb.class_id != 1) continue;

    cv::Point2f ground_xy;
    if (!acc::TryGetGroundBottomCenterXY(tb, ground_xy)) continue;

    if (ground_xy.x < cfg_.roi_x_min_m || ground_xy.x > cfg_.roi_x_max_m) continue;
    if (std::abs(ground_xy.y) > cfg_.roi_y_half_m) continue;

    seen[tb.id] = true;

    // classify-based warning
    if (cfg_.enable_classify_warning) {
      const int cls_id = get_classify_id(tb, ground_xy);
      if (cls_id >= 9 && cls_id <= 12) {
        if (!classify_warn_raw || ground_xy.x < classify_best_x) {
          classify_warn_raw = true;
          classify_best_x = ground_xy.x;
          classify_warn_track_id = tb.id;
          classify_warn_cls_id = cls_id;
          classify_warn_pos = ground_xy;
        }
      }
    }

    const auto st = tracker_.Update(tb.id, tb.frame, ground_xy, dt_s);

    Trajectory traj;
    traj.id = tb.id;
    traj.age_frames = st.age;
    traj.detection_score = tb.score;
    traj.p0 = st.pos;
    traj.vrel = st.vel;
    traj.heading_valid = tb.target_heading_valid;
    traj.heading_deg   = tb.target_heading_deg;

    // --- IEEE T-ITS Revision: Geometry-Guided Prediction Logic (INSERT THIS) ---
    const float T  = std::max(cfg_.horizon_s, 0.5f);
    const float dt = std::max(cfg_.step_s, 0.05f);
    const int   N  = static_cast<int>(std::floor(T / dt)) + 1;

    // [Step 1] 解析慣性狀態 (Inertial State)
    cv::Point2f vel_inertial = st.vel;
    float speed = cv::norm(vel_inertial);
    float angle_inertial = std::atan2(vel_inertial.y, vel_inertial.x);

    // [Step 2] 融合權重設定 (Fusion Configuration)
    // 如果骨架角度無效，則 alpha = 0 (完全退回慣性導航)
    // 如果有效，alpha = 0.4 (引入 40% 的骨架意圖)
    float alpha = 0.0f;
    float angle_skeleton = angle_inertial; // 預設同慣性

    if (tb.target_heading_valid) {
        alpha = std::clamp(cfg_.heading_fusion_alpha, 0.0f, 1.0f);
        // 假設 heading_deg 為度數，需轉弧度。注意 OpenCV 坐標系方向
        angle_skeleton = tb.target_heading_deg * CV_PI / 180.0f;
    }

    // [Step 3] 向量融合 (Vector Fusion)
    // 使用向量合成避免角度跳變問題 (-180 vs 180)
    float fuse_cos = (1.0f - alpha) * std::cos(angle_inertial) + alpha * std::cos(angle_skeleton);
    float fuse_sin = (1.0f - alpha) * std::sin(angle_inertial) + alpha * std::sin(angle_skeleton);
    float angle_fused = std::atan2(fuse_sin, fuse_cos);

    // 重組預測速度向量 (保留慣性速率，改變方向)
    cv::Point2f vel_pred(speed * std::cos(angle_fused), speed * std::sin(angle_fused));

    // [Step 4] 生成修正後的軌跡 (Trajectory Generation)
    traj.pred.reserve((size_t)N);
    for (int i = 0; i <= N; ++i) {
      const float t = (float)i * dt;
      // 使用融合後的 vel_pred 進行推估
      traj.pred.emplace_back(st.pos + vel_pred * t);
    }
    // --------------------------------------------------------------------------

    out.all_trajs.push_back(std::move(traj));
  }

  tracker_.MarkMissedAndPrune(seen);

  // prune classify cache
  if (frame_now >= 0 && cfg_.classify_ttl_frames > 0 && !classify_by_id_.empty()) {
    for (auto it = classify_by_id_.begin(); it != classify_by_id_.end(); ) {
      const int id = it->first;
      const bool is_seen = (seen.find(id) != seen.end()) && seen.at(id);
      const bool too_old = (frame_now - it->second.last_frame) > cfg_.classify_ttl_frames;
      if (!is_seen && too_old) it = classify_by_id_.erase(it);
      else ++it;
    }
  }

  struct Cand {
    size_t idx = 0;
    int id = -1;
    risk::ThreatEval ev;
  };

  std::vector<Cand> cands;
  cands.reserve(out.all_trajs.size());

  for (size_t i = 0; i < out.all_trajs.size(); ++i) {
    const auto& tr = out.all_trajs[i];

    const float dist_now = risk::MinDistToPath(tr.p0, out.ego_path);
    const bool allow_early_close = (dist_now <= cfg_.corridor_half_width_m) && (tr.p0.x >= 0.f) && (tr.p0.x <= cfg_.dis_brake_m);

    if (tr.age_frames < cfg_.track_warmup_frames && !allow_early_close) continue;

    const float min_app = allow_early_close ? 0.0f : cfg_.min_approach_speed_mps;

    auto ev = risk::EvaluateConstantVelocityCorridorRisk(
        tr.p0, tr.vrel, tr.pred, out.ego_path,
        cfg_.danger_forward_m,
        cfg_.corridor_half_width_m,
        cfg_.attention_half_width_m,
        min_app,
        std::max(cfg_.step_s, 0.05f));

    if (!ev.valid) continue;

    Cand c;
    c.idx = i;
    c.id = tr.id;
    c.ev = std::move(ev);
    cands.push_back(std::move(c));
  }

  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& b) { return a.ev.score < b.ev.score; });

  auto find_by_id = [&](int id) -> const Cand* {
    for (const auto& c : cands) if (c.id == id) return &c;
    return nullptr;
  };

  const Cand* best = cands.empty() ? nullptr : &cands.front();
  const Cand* chosen = best;

  // threat hysteresis
  if (hold_left_ > 0 && last_threat_id_ >= 0) {
    const Cand* last = find_by_id(last_threat_id_);
    if (last) {
      if (!best || last->ev.t_hit_s <= best->ev.t_hit_s + cfg_.threat_switch_hysteresis_s) {
        chosen = last;
      }
    }
  }

  if (chosen) {
    const auto& tr = out.all_trajs[chosen->idx];

    out.threat_id = chosen->id;
    out.threat_ttc_s = chosen->ev.t_hit_s;
    out.threat_min_dist_m = chosen->ev.min_dist_m;
    out.threat_dist_now_m = chosen->ev.dist_now_m;
    out.threat_approach_speed_mps = chosen->ev.approach_speed_mps;
    out.threat_score = chosen->ev.score;

    out.threat_pos = chosen->ev.hit_pos;
    out.threat_vrel = tr.vrel;
    out.threat_traj = tr;
  }

  // collision warning policy (ONLY based on chosen collision threat)
  bool collision_warn_raw = false;
  if (chosen && out.threat_id >= 0) {
    collision_warn_raw =
        (out.threat_ttc_s <= cfg_.ttc_warn_s) ||
        (out.threat_min_dist_m <= cfg_.corridor_half_width_m);
  }

  if (collision_warn_raw && classify_warn_raw && classify_warn_track_id == out.threat_id) {
    if (classify_warn_cls_id >= 9 && classify_warn_cls_id <= 11) classify_warn_cls_id = 12;
  }

  const bool publish_warn_raw = collision_warn_raw || (cfg_.enable_classify_warning && classify_warn_raw);
  out.collision_warning_raw = collision_warn_raw;
  out.classify_warning_raw = classify_warn_raw;

  bool publish_warn = publish_warn_raw;
  if (cfg_.enable_warning_kf) {
    publish_warn = warning_filter_.Update(
        publish_warn_raw, dt_s,
        cfg_.warning_kf_q_per_s,
        cfg_.warning_kf_r,
        cfg_.warning_kf_on_th,
        cfg_.warning_kf_off_th,
        &warning_kf_x_);
  }
  out.warning = publish_warn;

  // classify-only warning: set threat_id for UI (does NOT affect collision_warn_raw)
  if (out.threat_id < 0 && classify_warn_raw && classify_warn_track_id >= 0) {
    out.threat_id = classify_warn_track_id;
    out.threat_pos = classify_warn_pos;
    out.threat_dist_now_m = risk::MinDistToPath(classify_warn_pos, out.ego_path);
    out.threat_min_dist_m = out.threat_dist_now_m;
    out.threat_ttc_s = std::numeric_limits<float>::infinity();
    out.threat_approach_speed_mps = 0.f;
    out.threat_score = std::numeric_limits<float>::infinity();
  }

  // update hysteresis state
  if (out.warning && out.threat_id >= 0) {
    last_threat_id_ = out.threat_id;
    hold_left_ = cfg_.threat_hold_frames;
  } else {
    if (hold_left_ > 0) hold_left_--;
    if (hold_left_ == 0) last_threat_id_ = -1;
  }
  out.threat_hold_left = hold_left_;

  // actuation: ONLY collision_warn_raw
  if (enable_actuation && collision_warn_raw && out.threat_id >= 0 && std::isfinite(out.threat_ttc_s)) {
    const float ttc = out.threat_ttc_s;
    const float dis = out.threat_pos.x;

    const float ttc_factor = Clamp01( (ttc - cfg_.ttc_brake_s) / std::max(1e-3f, (cfg_.ttc_warn_s - cfg_.ttc_brake_s)) );
    const float dis_factor = Clamp01( (dis - cfg_.dis_brake_m) / std::max(1e-3f, (cfg_.dis_warn_m - cfg_.dis_brake_m)) );

    const float alpha = std::min(ttc_factor, dis_factor);

    const float base_speed = std::max(0.f, *io_speed_kmh);
    const float new_speed  = std::min(base_speed, base_speed * alpha);
    *io_speed_kmh = new_speed;

    const float brake_add = (1.0f - alpha) * cfg_.max_extra_brake_0_10;
    *io_brake_0_10 = std::max(*io_brake_0_10, brake_add);

    const float steer_offset = (out.threat_pos.y >= 0.f ? -1.f : +1.f) * cfg_.max_avoid_steer_deg * (1.0f - alpha);
    *io_steer_deg = *io_steer_deg + steer_offset;
  }

  // debug
  {
    std::ostringstream oss;
    oss << "collision_assist:"
        << " warn=" << (out.warning ? 1 : 0)
        << " threat_id=" << out.threat_id
        << " ttc=" << out.threat_ttc_s
        << " min_dist=" << out.threat_min_dist_m
        << " dist_now=" << out.threat_dist_now_m
        << " app=" << out.threat_approach_speed_mps
        << " vrel=(" << out.threat_vrel.x << "," << out.threat_vrel.y << ")"
        << " hold=" << out.threat_hold_left
        << " act=" << (enable_actuation ? "true" : "false")
        << " coll_raw=" << (collision_warn_raw ? 1 : 0)
        << " cls_raw=" << (classify_warn_raw ? 1 : 0)
        << " cls_id=" << classify_warn_cls_id
        << " cls_tid=" << classify_warn_track_id
        << " pub_raw=" << (publish_warn_raw ? 1 : 0)
        << " kf_x=" << warning_kf_x_
        << " warn=" << (out.warning ? 1 : 0);

    oss << " cand=[";
    const size_t K = std::min<size_t>(3, cands.size());
    for (size_t k = 0; k < K; ++k) {
      const auto& c = cands[k];
      oss << "{id=" << c.id
          << " t=" << c.ev.t_hit_s
          << " d=" << c.ev.min_dist_m
          << " now=" << c.ev.dist_now_m
          << " app=" << c.ev.approach_speed_mps
          << " sc=" << c.ev.score
          << "}";
      if (k + 1 < K) oss << ",";
    }
    oss << "]";
    out.debug = oss.str();
  }

  return out;
}

CollisionAssistOutput CollisionAssist::Step(const std::vector<TrackingBox>& world_result,
                                            float ego_speed_mps,
                                            double steer_cmd_deg,
                                            float dt_s,
                                            bool enable_actuation,
                                            float* io_speed_kmh,
                                            double* io_steer_deg,
                                            double* io_brake_0_10)
{
  CollisionAssistOutput out;
  if (!io_speed_kmh || !io_steer_deg || !io_brake_0_10) {
    out.debug = "io_speed_kmh/io_steer_deg/io_brake_0_10 is null";
    return out;
  }

  float speed_f = *io_speed_kmh;
  float steer_f = static_cast<float>(*io_steer_deg);
  float brake_f = static_cast<float>(*io_brake_0_10);

  out = this->Step(
      world_result,
      ego_speed_mps,
      static_cast<float>(steer_cmd_deg),
      dt_s,
      enable_actuation,
      &speed_f, &steer_f, &brake_f);

  *io_speed_kmh  = speed_f;
  *io_steer_deg  = static_cast<double>(steer_f);
  *io_brake_0_10 = static_cast<double>(brake_f);
  return out;
}

CollisionAssistOutput CollisionAssist::Step(const std::vector<TrackingBox>& world_result,
                                            double ego_speed_mps,
                                            double steer_cmd_deg,
                                            double dt_s,
                                            bool enable_actuation,
                                            double* io_speed_kmh,
                                            double* io_steer_deg,
                                            double* io_brake_0_10)
{
  CollisionAssistOutput out;
  if (!io_speed_kmh || !io_steer_deg || !io_brake_0_10) {
    out.debug = "io_speed_kmh/io_steer_deg/io_brake_0_10 is null";
    return out;
  }

  float speed_f = static_cast<float>(*io_speed_kmh);
  float steer_f = static_cast<float>(*io_steer_deg);
  float brake_f = static_cast<float>(*io_brake_0_10);

  out = this->Step(
      world_result,
      static_cast<float>(ego_speed_mps),
      static_cast<float>(steer_cmd_deg),
      static_cast<float>(dt_s),
      enable_actuation,
      &speed_f, &steer_f, &brake_f);

  *io_speed_kmh  = static_cast<double>(speed_f);
  *io_steer_deg  = static_cast<double>(steer_f);
  *io_brake_0_10 = static_cast<double>(brake_f);
  return out;
}

} // namespace collision
