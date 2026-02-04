#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <limits>
#include <unordered_map>
#include <opencv2/core.hpp>

#include "SortTracking.h"
#include "GeometryAdapter.h"
#include "VehiclePhysics.h"
#include "VehicleControlApi.h"

#include "ObjMotionTracker2D.h"
#include "WarningFilter.h"

namespace collision {

struct Trajectory {
  int id = -1;
  int age_frames = 0;                 // 速度估測成熟度 (warmup)
  cv::Point2f p0{};
  cv::Point2f vrel{};
  bool heading_valid = false;
  float heading_deg = 0.f;
  std::vector<cv::Point2f> pred;
};

struct CollisionAssistConfig {
  float roi_y_half_m = 5.63f;
  float roi_x_min_m  = 0.0f;
  float roi_x_max_m  = 60.0f;

  float danger_forward_m       = 10.0f;
  float corridor_half_width_m  = 1.2f;
  float path_sample_step_m     = 0.25f;

  float horizon_s = 4.0f;
  float step_s    = 0.2f;

  float ttc_warn_s  = 3.0f;
  float ttc_brake_s = 1.2f;
  float dis_warn_m  = 12.0f;
  float dis_brake_m = 6.0f;

  float max_extra_brake_0_10 = 4.0f;
  float max_avoid_steer_deg  = 4.0f;

  // ----------------------------------------------------------------------
  // Classify-based CA warning (class_name_classify id 9~12)
  // - 建議直接使用 TrackingBox.traffic_class_num (你已經把它塞進 TrackingBox)
  // - UpdateClassify() 僅做備援：如果你有獨立的 classify callback。
  // - bbox fallback 很容易造成 ca.warning 常駐=1，預設關閉。
  bool enable_classify_warning = true;

  // 0) Prefer tb.traffic_class_num (9~12 才有效)
  bool prefer_trackingbox_classify = true;

  // 1) Optional cache for UpdateClassify()
  int  classify_ttl_frames = 5;

  // 2) bbox fallback (NOT recommended)
  bool enable_bbox_fallback_classify = false;

  // bbox fallback：沿用你 run_traffic_classification 的 car ROI 條件
  int classify_car_roi_x_min_px = 400;
  int classify_car_roi_x_max_px = 880;
  int classify_car_roi_y_min_px = 250;
  int classify_center_x_px      = 640; // 1280 寬畫面中心
  int classify_lr_deadband_px   = 80;  // 左/中/右分界

  // ----------------------------------------------------------------------
  // Kalman filter for published ca.warning
  bool  enable_warning_kf     = true;
  float warning_kf_q_per_s    = 2.0f;  // 越大 -> 跟隨越快
  float warning_kf_r          = 1.0f;  // 越大 -> 更不信任量測(更平滑)
  float warning_kf_on_th      = 0.65f; // 觸發門檻
  float warning_kf_off_th     = 0.35f; // 解除門檻

  // --- Anti-false-positive gates ---
  int   track_warmup_frames = 3;          // 速度要成熟
  float attention_half_width_m = 2.5f;    // 關注帶 (比 corridor 寬)
  float min_approach_speed_mps = 0.8f;    // 接近速度門檻

  // Threat 維持：讓 threat_id 穩定，便於知道哪台觸發
  int   threat_hold_frames = 6;
  float threat_switch_hysteresis_s = 0.4f;
};

struct CollisionAssistOutput {
  bool warning = false;
  int threat_id = -1;

  float threat_ttc_s = std::numeric_limits<float>::infinity();
  float threat_min_dist_m = std::numeric_limits<float>::infinity();
  float threat_dist_now_m = std::numeric_limits<float>::infinity();
  float threat_approach_speed_mps = 0.f;
  float threat_score = std::numeric_limits<float>::infinity();

  int threat_hold_left = 0;

  cv::Point2f threat_pos{};
  cv::Point2f threat_vrel{};

  std::vector<cv::Point2f> ego_path;
  Trajectory threat_traj;
  std::vector<Trajectory> all_trajs;

  std::string debug;
};

class CollisionAssist {
public:
  explicit CollisionAssist(const CollisionAssistConfig& cfg = {}) : cfg_(cfg) {}

  // 若你外部仍在用分類器結果 callback，可用這個更新 cache。
  void UpdateClassify(int track_id, int frame, int classify_id);

  void Reset() {
    tracker_.Reset();
    classify_by_id_.clear();
    warning_filter_.Reset();
    warning_kf_x_ = 0.f;
    last_threat_id_ = -1;
    hold_left_ = 0;
  }

  CollisionAssistOutput Step(const std::vector<TrackingBox>& world_result,
                            float ego_speed_mps,
                            float steer_cmd_deg,
                            float dt_s,
                            bool enable_actuation,
                            float* io_speed_kmh,
                            float* io_steer_deg,
                            float* io_brake_0_10);

  CollisionAssistOutput Step(const std::vector<TrackingBox>& world_result,
                            float ego_speed_mps,
                            double steer_cmd_deg,
                            float dt_s,
                            bool enable_actuation,
                            float* io_speed_kmh,
                            double* io_steer_deg,
                            double* io_brake_0_10);

  CollisionAssistOutput Step(const std::vector<TrackingBox>& world_result,
                            double ego_speed_mps,
                            double steer_cmd_deg,
                            double dt_s,
                            bool enable_actuation,
                            double* io_speed_kmh,
                            double* io_steer_deg,
                            double* io_brake_0_10);

private:
  CollisionAssistConfig cfg_;
  AlphaBetaTracker2D tracker_;

  struct ClassifyState { int cls = -1; int last_frame = -1; };
  std::unordered_map<int, ClassifyState> classify_by_id_;

  WarningFilter warning_filter_;
  float warning_kf_x_ = 0.f;

  int last_threat_id_ = -1;
  int hold_left_ = 0;

  static std::vector<cv::Point2f> SampleEgoPath(float steer_cmd_deg,
                                                float forward_m,
                                                float step_m);

  static float MinDistToPath(const cv::Point2f& p,
                             const std::vector<cv::Point2f>& path);

  static float Clamp01(float x) { return std::max(0.f, std::min(1.f, x)); }
};

} // namespace collision
