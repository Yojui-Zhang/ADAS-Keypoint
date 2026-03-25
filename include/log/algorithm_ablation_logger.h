#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "VehicleControlApi.h"
#include "config.h"

namespace ablation {

struct AlgorithmAblationOptions {
  bool enabled = true;                              // ablation logger 總開關
  std::string output_path;                          // 指定完整 CSV 輸出路徑
  std::string output_dir = "research_logs";         // 自動輸出時使用的資料夾
  double steering_ratio = 14.5;                     // 方向盤角/路輪角轉向比
  double wheelbase_m = 2.62;                        // 軸距（m）
  int flush_every_n = 30;                           // 每 N 幀 flush 一次
  int plot_size_px = 1200;                          // 路徑圖邊長（像素）
  int plot_margin_px = 80;                          // 路徑圖邊界留白（像素）

  bool virtual_road_enable = false;                 // 是否啟用 reference road 比對
  std::string virtual_road_mode = "straight";       // straight | arc | s_curve | csv
  std::string virtual_road_csv_path;                // mode=csv 時的 x,y 檔案路徑
  double virtual_road_length_m = 300.0;             // 內建道路長度（m）
  double virtual_road_step_m = 0.5;                 // 內建道路取樣間距（m）
  double virtual_road_lane_width_m = 3.5;           // 車道寬（m），用於偏離判定
  double virtual_road_arc_radius_m = 120.0;         // arc 半徑（m）
  double virtual_road_s_amplitude_m = 2.0;          // s_curve 振幅（m）
  double virtual_road_s_wavelength_m = 80.0;        // s_curve 波長（m）
};

struct VirtualRoadSimulationOptions {
  uint64_t frame_count = 1200;             // 模擬總幀數
  double dt_s = 0.05;                      // 模擬時間步長（s）
  double speed_kmh = 30.0;                 // 模擬固定車速（km/h）
  double max_steer_deg = 35.0;             // 轉角飽和上限（deg）

  // VC on 控制增益（較強，應更貼近 reference road）
  double vc_k_cte = 6.0;                   // CTE 控制增益（deg/m）
  double vc_k_heading = 0.9;               // heading 控制增益（deg/deg）

  // VC off 控制增益（較弱，並加入偏置與擾動）
  double raw_k_cte = 1.8;                  // CTE 控制增益（deg/m）
  double raw_k_heading = 0.2;              // heading 控制增益（deg/deg）
  double raw_steer_bias_deg = 2.5;         // 固定轉向偏置（deg）
  double raw_steer_osc_amp_deg = 1.0;      // 週期擾動振幅（deg）
  double raw_steer_osc_period_s = 6.0;     // 週期擾動週期（s）
};

struct AlgorithmAblationFrame {
  uint64_t frame_index = 0;                         // 幀序號
  uint64_t frame_sync_ns = 0;                       // 時間同步戳（ns）
  double dt_s = 0.0;                                // 兩幀間隔（s）
  double ego_speed_kmh = 0.0;                       // 自車速度（km/h）

  const std::vector<TrackingBox>* world_before_skeleton = nullptr;  // RunVehicleSkeletonAndHeading 前
  const std::vector<TrackingBox>* world_after_skeleton = nullptr;   // RunVehicleSkeletonAndHeading 後

  stability::VehicleControlCommand vehicle_control_cmd{};            // VehicleControl_Run 輸出命令
};

struct AlgorithmAblationResult {
  int skeleton_vehicle_count = 0;                    // 參與統計的車輛數
  int skeleton_before_valid_count = 0;               // 處理前有效 heading 數
  int skeleton_after_valid_count = 0;                // 處理後有效 heading 數
  int skeleton_changed_count = 0;                    // heading 被更新的車輛數
  double skeleton_heading_mean_abs_delta_deg = 0.0;  // heading 平均絕對變化（deg）
  double skeleton_heading_max_abs_delta_deg = 0.0;   // heading 最大絕對變化（deg）

  double vc_raw_speed_diff_kmh = 0.0;                // VC on/off 速度差（km/h）
  double vc_raw_steer_diff_deg = 0.0;                // VC on/off 轉角差（deg）
  double vc_raw_brake_diff_0_10 = 0.0;               // VC on/off 煞車差（0~10）
  double route_gap_m = 0.0;                          // 兩條積分路徑間距（m）

  bool virtual_road_valid = false;                   // 本幀是否可投影到 reference road
  double vc_cte_m = 0.0;                             // VC on CTE（m）
  double vc_heading_err_deg = 0.0;                   // VC on 航向誤差（deg）
  int vc_lane_departure = 0;                         // VC on 是否偏離車道（0/1）
  double raw_cte_m = 0.0;                            // VC off CTE（m）
  double raw_heading_err_deg = 0.0;                  // VC off 航向誤差（deg）
  int raw_lane_departure = 0;                        // VC off 是否偏離車道（0/1）
};

class AlgorithmAblationLogger {
public:
  explicit AlgorithmAblationLogger(const AlgorithmAblationOptions& options = {});
  ~AlgorithmAblationLogger();

  bool Start(std::string* out_error = nullptr);
  AlgorithmAblationResult Step(const AlgorithmAblationFrame& frame);
  bool RunVirtualRoadSimulation(const VirtualRoadSimulationOptions& sim,
                                std::string* out_error = nullptr);
  void Stop();

  bool IsEnabled() const { return options_.enabled; }
  bool IsRunning() const { return running_; }
  const std::string& OutputPath() const { return output_path_; }

private:
  struct PoseState {
    double x_m = 0.0;
    double y_m = 0.0;
    double heading_rad = 0.0;
    double distance_m = 0.0;
  };

  struct Summary {
    uint64_t sample_count = 0;
    uint64_t skeleton_changed_frames = 0;

    double sum_skeleton_heading_mean_abs_delta_deg = 0.0;
    double max_skeleton_heading_abs_delta_deg = 0.0;

    double sum_abs_speed_diff_kmh = 0.0;
    double max_abs_speed_diff_kmh = 0.0;
    double sum_abs_steer_diff_deg = 0.0;
    double max_abs_steer_diff_deg = 0.0;
    double sum_abs_brake_diff_0_10 = 0.0;
    double max_abs_brake_diff_0_10 = 0.0;

    double max_route_gap_m = 0.0;
    double final_route_gap_m = 0.0;

    uint64_t virtual_road_valid_frames = 0;
    uint64_t vc_lane_departure_count = 0;
    uint64_t raw_lane_departure_count = 0;
    double sum_abs_vc_cte_m = 0.0;
    double max_abs_vc_cte_m = 0.0;
    double sum_abs_raw_cte_m = 0.0;
    double max_abs_raw_cte_m = 0.0;
    double sum_abs_vc_heading_err_deg = 0.0;
    double max_abs_vc_heading_err_deg = 0.0;
    double sum_abs_raw_heading_err_deg = 0.0;
    double max_abs_raw_heading_err_deg = 0.0;
  };

  void WriteHeader();
  void WriteSummaryFile();
  void WriteRoutePlot();
  bool InitVirtualRoad(std::string* out_error);
  bool ParseEnvDouble(const char* key, double* out_value);
  void UpdatePose(PoseState* pose, double speed_kmh, double steer_deg, double dt_s) const;
  std::string ResolveOutputPath() const;

  static bool ParseEnvBool(const char* value, bool default_value);
  static std::string ToLower(std::string s);
  static double Clamp(double x, double lo, double hi);
  static double WrapPi(double rad);
  static double DegToRad(double deg);
  static double RadToDeg(double rad);
  static double NormalizeDeltaDeg(double deg);
  static double FiniteOrNaN(double value);

  AlgorithmAblationOptions options_;
  bool running_ = false;
  std::string output_path_;
  std::ofstream out_;
  uint64_t flush_counter_ = 0;
  std::chrono::steady_clock::time_point start_steady_;

  PoseState route_vc_;
  PoseState route_raw_;
  std::vector<cv::Point2d> path_vc_;
  std::vector<cv::Point2d> path_raw_;
  bool virtual_road_active_ = false;
  std::string virtual_road_mode_used_;
  std::vector<cv::Point2d> virtual_road_path_;
  Summary summary_;
};

}  // namespace ablation
