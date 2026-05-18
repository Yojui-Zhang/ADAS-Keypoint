#include "algorithm_ablation_logger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

bool IsFinite(double v) {
  return std::isfinite(v);
}

std::string TimestampStringNow() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);

  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif

  char buf[32] = {0};
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
  return std::string(buf);
}

std::string ParentDirOf(const std::string& p) {
  std::filesystem::path path(p);
  return path.has_parent_path() ? path.parent_path().string() : std::string();
}

bool ParseCsvXYLine(const std::string& line, double* out_x, double* out_y) {
  if (!out_x || !out_y) return false;

  std::string s = line;
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
  if (s.empty()) return false;
  if (s[0] == '#') return false;
  std::replace(s.begin(), s.end(), ';', ',');

  int found = 0;
  double values[2] = {0.0, 0.0};
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(token.begin(),
                std::find_if(token.begin(),
                             token.end(),
                             [](unsigned char c) { return !std::isspace(c); }));
    token.erase(std::find_if(token.rbegin(),
                             token.rend(),
                             [](unsigned char c) { return !std::isspace(c); }).base(),
                token.end());
    if (token.empty()) continue;

    char* endptr = nullptr;
    const double v = std::strtod(token.c_str(), &endptr);
    if (endptr == token.c_str()) continue;
    while (*endptr && std::isspace(static_cast<unsigned char>(*endptr))) ++endptr;
    if (*endptr != '\0') continue;

    values[found++] = v;
    if (found >= 2) break;
  }

  if (found < 2) return false;
  *out_x = values[0];
  *out_y = values[1];
  return true;
}

struct SkeletonDiff {
  int vehicle_count = 0;
  int before_valid_count = 0;
  int after_valid_count = 0;
  int changed_count = 0;
  double mean_abs_delta_deg = 0.0;
  double max_abs_delta_deg = 0.0;
};

struct VirtualRoadProjection {
  bool valid = false;
  double cte_m = 0.0;
  double heading_err_deg = 0.0;
  size_t segment_index = 0;
  double ref_heading_rad = 0.0;
  double ref_curvature_m_inv = 0.0;
};

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

double Dot3(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 Add3(const Vec3& a, const Vec3& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 Sub3(const Vec3& a, const Vec3& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 Scale3(const Vec3& a, double scale) {
  return {a[0] * scale, a[1] * scale, a[2] * scale};
}

Vec3 MatVec3(const Mat3& a, const Vec3& x) {
  Vec3 out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2];
  }
  return out;
}

Mat3 MatMul3(const Mat3& a, const Mat3& b) {
  Mat3 out{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      double acc = 0.0;
      for (size_t k = 0; k < 3; ++k) {
        acc += a[i][k] * b[k][j];
      }
      out[i][j] = acc;
    }
  }
  return out;
}

Mat3 Transpose3(const Mat3& a) {
  Mat3 out{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      out[i][j] = a[j][i];
    }
  }
  return out;
}

Mat3 AddMat3(const Mat3& a, const Mat3& b) {
  Mat3 out{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      out[i][j] = a[i][j] + b[i][j];
    }
  }
  return out;
}

Mat3 ScaleMat3(const Mat3& a, double scale) {
  Mat3 out{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      out[i][j] = a[i][j] * scale;
    }
  }
  return out;
}

Mat3 Outer3(const Vec3& a, const Vec3& b) {
  Mat3 out{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      out[i][j] = a[i] * b[j];
    }
  }
  return out;
}

double EstimatePathCurvature(const std::vector<cv::Point2d>& path, size_t segment_index) {
  if (path.size() < 3) return 0.0;

  const size_t i1 = std::min(segment_index, path.size() - 1);
  const size_t i0 = (i1 > 0) ? (i1 - 1) : i1;
  const size_t i2 = std::min(i1 + 1, path.size() - 1);
  if (i0 == i1 || i1 == i2) return 0.0;

  const cv::Point2d a = path[i1] - path[i0];
  const cv::Point2d b = path[i2] - path[i1];
  const cv::Point2d c = path[i2] - path[i0];
  const double len_a = std::sqrt(a.dot(a));
  const double len_b = std::sqrt(b.dot(b));
  const double len_c = std::sqrt(c.dot(c));
  if (len_a < 1e-9 || len_b < 1e-9 || len_c < 1e-9) return 0.0;

  const double cross = a.x * b.y - a.y * b.x;
  return (2.0 * cross) / (len_a * len_b * len_c);
}

double ComputePreviewMpcSteerDeg(const VirtualRoadProjection& proj,
                                 double speed_kmh,
                                 double dt_s,
                                 double wheelbase_m,
                                 double steering_ratio,
                                 double max_steer_deg,
                                 double last_steer_deg,
                                 const ablation::VirtualRoadSimulationOptions& cfg) {
  if (!proj.valid || !cfg.preview_mpc_enable) {
    return 0.0;
  }

  const double v = std::max(0.0, speed_kmh) / 3.6;
  const double dt = std::max(1e-3, dt_s);
  const double wheelbase = std::max(1e-3, wheelbase_m);
  const double steer_ratio = std::max(1e-3, steering_ratio);
  const double max_road_wheel_rad =
      (max_steer_deg / steer_ratio) * (3.14159265358979323846 / 180.0);
  const double max_rate_rad =
      (cfg.preview_mpc_max_steer_rate_deg_s / steer_ratio) *
      (3.14159265358979323846 / 180.0) * dt;

  const double heading_err_rad = proj.heading_err_deg * (3.14159265358979323846 / 180.0);
  const double last_delta_rad =
      (last_steer_deg / steer_ratio) * (3.14159265358979323846 / 180.0);
  const double curvature = proj.ref_curvature_m_inv;
  const double v_gain = (v * dt) / wheelbase;

  const Mat3 A{{
      {{1.0, v * dt, 0.0}},
      {{0.0, 1.0, v_gain}},
      {{0.0, 0.0, 1.0}},
  }};
  const Vec3 B{0.0, v_gain, 1.0};
  const Vec3 c{0.0, -v * curvature * dt, 0.0};

  Mat3 Q{};
  Q[0][0] = cfg.preview_mpc_q_cte;
  Q[1][1] = cfg.preview_mpc_q_heading;
  Q[2][2] = cfg.preview_mpc_q_steer;

  Mat3 P = Q;
  P[0][0] *= 4.0;
  P[1][1] *= 4.0;
  P[2][2] *= 2.0;
  Vec3 s{0.0, 0.0, 0.0};
  const double R = std::max(1e-6, cfg.preview_mpc_r_steer_rate);

  Vec3 first_K{0.0, 0.0, 0.0};
  double first_k = 0.0;
  const size_t horizon = std::max<size_t>(1, cfg.preview_mpc_horizon);

  for (size_t step = 0; step < horizon; ++step) {
    const Mat3 P_prev = P;
    const Vec3 s_prev = s;
    const Vec3 PB = MatVec3(P_prev, B);
    const double G = R + Dot3(B, PB);
    if (G < 1e-9) break;

    const Mat3 PA = MatMul3(P_prev, A);
    Vec3 BtPA{};
    for (size_t j = 0; j < 3; ++j) {
      BtPA[j] = B[0] * PA[0][j] + B[1] * PA[1][j] + B[2] * PA[2][j];
    }
    const Vec3 K = Scale3(BtPA, 1.0 / G);

    const Vec3 Pc_plus_s = Add3(MatVec3(P_prev, c), s_prev);
    const double k_ff = Dot3(B, Pc_plus_s) / G;

    if (step == 0) {
      first_K = K;
      first_k = k_ff;
    }

    const Mat3 A_t = Transpose3(A);
    const Mat3 A_t_P_A = MatMul3(A_t, PA);
    const Mat3 A_t_P_B_K = Outer3(MatVec3(A_t, PB), K);
    P = AddMat3(Q, AddMat3(A_t_P_A, ScaleMat3(A_t_P_B_K, -1.0)));

    const Vec3 P_B_k = Scale3(PB, k_ff);
    const Vec3 s_term = Sub3(Pc_plus_s, P_B_k);
    s = MatVec3(A_t, s_term);
  }

  const Vec3 z0{proj.cte_m, heading_err_rad, last_delta_rad};
  double delta_u_rad = -(Dot3(first_K, z0) + first_k);
  delta_u_rad = std::max(-max_rate_rad, std::min(max_rate_rad, delta_u_rad));

  double delta_cmd_rad = last_delta_rad + delta_u_rad;
  delta_cmd_rad = std::max(-max_road_wheel_rad, std::min(max_road_wheel_rad, delta_cmd_rad));
  return delta_cmd_rad * steer_ratio * (180.0 / 3.14159265358979323846);
}

double WrapPiLocal(double rad) {
  constexpr double kPi = 3.14159265358979323846;
  while (rad > kPi) rad -= 2.0 * kPi;
  while (rad < -kPi) rad += 2.0 * kPi;
  return rad;
}

double RadToDegLocal(double rad) {
  return rad * (180.0 / 3.14159265358979323846);
}

VirtualRoadProjection ProjectPoseToPath(const cv::Point2d& pos,
                                        double heading_rad,
                                        const std::vector<cv::Point2d>& path) {
  VirtualRoadProjection out;
  if (path.size() < 2) return out;

  double best_d2 = std::numeric_limits<double>::infinity();

  for (size_t i = 1; i < path.size(); ++i) {
    const cv::Point2d p0 = path[i - 1];
    const cv::Point2d p1 = path[i];
    const cv::Point2d d = p1 - p0;
    const double len2 = d.dot(d);
    if (len2 < 1e-12) continue;

    const double t_raw = (pos - p0).dot(d) / len2;
    const double t = std::max(0.0, std::min(1.0, t_raw));
    const cv::Point2d q = p0 + d * t;
    const cv::Point2d diff = pos - q;
    const double d2 = diff.dot(diff);

    if (d2 >= best_d2) continue;
    best_d2 = d2;

    const double seg_len = std::sqrt(len2);
    const double cross = d.x * diff.y - d.y * diff.x;
    const double cte_signed = (seg_len > 1e-9) ? (cross / seg_len) : 0.0;
    const double ref_heading = std::atan2(d.y, d.x);
    const double heading_err_deg = RadToDegLocal(WrapPiLocal(heading_rad - ref_heading));

    out.valid = true;
    out.cte_m = cte_signed;
    out.heading_err_deg = heading_err_deg;
    out.segment_index = i;
    out.ref_heading_rad = ref_heading;
    out.ref_curvature_m_inv = EstimatePathCurvature(path, i);
  }

  return out;
}

}  // namespace

namespace ablation {

AlgorithmAblationLogger::AlgorithmAblationLogger(const AlgorithmAblationOptions& options)
    : options_(options) {}

AlgorithmAblationLogger::~AlgorithmAblationLogger() {
  Stop();
}

bool AlgorithmAblationLogger::ParseEnvBool(const char* value, bool default_value) {
  if (!value) return default_value;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  return default_value;
}

std::string AlgorithmAblationLogger::ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

double AlgorithmAblationLogger::Clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

double AlgorithmAblationLogger::DegToRad(double deg) {
  return deg * (3.14159265358979323846 / 180.0);
}

double AlgorithmAblationLogger::RadToDeg(double rad) {
  return rad * (180.0 / 3.14159265358979323846);
}

double AlgorithmAblationLogger::WrapPi(double rad) {
  constexpr double kPi = 3.14159265358979323846;
  while (rad > kPi) rad -= 2.0 * kPi;
  while (rad < -kPi) rad += 2.0 * kPi;
  return rad;
}

double AlgorithmAblationLogger::NormalizeDeltaDeg(double deg) {
  while (deg > 180.0) deg -= 360.0;
  while (deg < -180.0) deg += 360.0;
  return deg;
}

double AlgorithmAblationLogger::FiniteOrNaN(double value) {
  if (IsFinite(value)) return value;
  return std::numeric_limits<double>::quiet_NaN();
}

bool AlgorithmAblationLogger::ParseEnvDouble(const char* key, double* out_value) {
  if (!out_value || !key) return false;
  const char* env = std::getenv(key);
  if (!env || !*env) return false;

  char* endptr = nullptr;
  const double v = std::strtod(env, &endptr);
  if (endptr == env) return false;
  while (*endptr && std::isspace(static_cast<unsigned char>(*endptr))) ++endptr;
  if (*endptr != '\0') return false;

  *out_value = v;
  return true;
}

bool AlgorithmAblationLogger::InitVirtualRoad(std::string* out_error) {
  virtual_road_active_ = false;
  virtual_road_mode_used_.clear();
  virtual_road_path_.clear();

  const char* env_enable = std::getenv("ADAS_ABLATION_VROAD_ENABLE");
  options_.virtual_road_enable = ParseEnvBool(env_enable, options_.virtual_road_enable);
  if (!options_.virtual_road_enable) return true;

  const char* env_mode = std::getenv("ADAS_ABLATION_VROAD_MODE");
  if (env_mode && *env_mode) options_.virtual_road_mode = std::string(env_mode);
  options_.virtual_road_mode = ToLower(options_.virtual_road_mode);

  const char* env_csv = std::getenv("ADAS_ABLATION_VROAD_FILE");
  if (env_csv && *env_csv) options_.virtual_road_csv_path = std::string(env_csv);

  ParseEnvDouble("ADAS_ABLATION_VROAD_LENGTH_M", &options_.virtual_road_length_m);
  ParseEnvDouble("ADAS_ABLATION_VROAD_STEP_M", &options_.virtual_road_step_m);
  ParseEnvDouble("ADAS_ABLATION_VROAD_LANE_WIDTH_M", &options_.virtual_road_lane_width_m);
  ParseEnvDouble("ADAS_ABLATION_VROAD_ARC_RADIUS_M", &options_.virtual_road_arc_radius_m);
  ParseEnvDouble("ADAS_ABLATION_VROAD_S_AMPLITUDE_M", &options_.virtual_road_s_amplitude_m);
  ParseEnvDouble("ADAS_ABLATION_VROAD_S_WAVELENGTH_M", &options_.virtual_road_s_wavelength_m);

  options_.virtual_road_length_m = std::max(5.0, options_.virtual_road_length_m);
  options_.virtual_road_step_m = Clamp(options_.virtual_road_step_m, 0.05, 5.0);
  options_.virtual_road_lane_width_m = Clamp(options_.virtual_road_lane_width_m, 0.5, 10.0);
  options_.virtual_road_s_wavelength_m = std::max(1.0, options_.virtual_road_s_wavelength_m);
  if (std::abs(options_.virtual_road_arc_radius_m) < 1.0) {
    options_.virtual_road_arc_radius_m = (options_.virtual_road_arc_radius_m >= 0.0) ? 120.0 : -120.0;
  }

  const std::string mode = options_.virtual_road_mode.empty() ? "straight" : options_.virtual_road_mode;

  if (mode == "straight") {
    for (double x = 0.0; x <= options_.virtual_road_length_m; x += options_.virtual_road_step_m) {
      virtual_road_path_.emplace_back(x, 0.0);
    }
  } else if (mode == "arc") {
    const double r = options_.virtual_road_arc_radius_m;
    const double r_abs = std::abs(r);
    const double theta_step = options_.virtual_road_step_m / r_abs;
    const double theta_max = options_.virtual_road_length_m / r_abs;
    const double sign_y = (r >= 0.0) ? 1.0 : -1.0;
    for (double theta = 0.0; theta <= theta_max; theta += theta_step) {
      const double x = r_abs * std::sin(theta);
      const double y = sign_y * r_abs * (1.0 - std::cos(theta));
      virtual_road_path_.emplace_back(x, y);
    }
  } else if (mode == "s_curve" || mode == "s-curve" || mode == "s") {
    const double amp = options_.virtual_road_s_amplitude_m;
    const double w = options_.virtual_road_s_wavelength_m;
    const double two_pi = 2.0 * 3.14159265358979323846;
    for (double x = 0.0; x <= options_.virtual_road_length_m; x += options_.virtual_road_step_m) {
      const double y = amp * std::sin(two_pi * x / w);
      virtual_road_path_.emplace_back(x, y);
    }
  } else if (mode == "csv") {
    if (options_.virtual_road_csv_path.empty()) {
      if (out_error) *out_error = "virtual road mode=csv but ADAS_ABLATION_VROAD_FILE is empty";
      return false;
    }

    std::ifstream in(options_.virtual_road_csv_path);
    if (!in.is_open()) {
      if (out_error) *out_error = "failed to open virtual road csv: " + options_.virtual_road_csv_path;
      return false;
    }

    std::string line;
    while (std::getline(in, line)) {
      double x = 0.0;
      double y = 0.0;
      if (ParseCsvXYLine(line, &x, &y)) {
        virtual_road_path_.emplace_back(x, y);
      }
    }
  } else {
    if (out_error) *out_error = "unknown virtual road mode: " + mode;
    return false;
  }

  if (virtual_road_path_.size() < 2) {
    if (out_error) *out_error = "virtual road has fewer than 2 points";
    return false;
  }

  virtual_road_active_ = true;
  virtual_road_mode_used_ = mode;
  return true;
}

std::string AlgorithmAblationLogger::ResolveOutputPath() const {
  const char* env_path = std::getenv("ADAS_ABLATION_LOG_PATH");
  if (env_path && *env_path) return std::string(env_path);

  const char* env_dir = std::getenv("ADAS_ABLATION_LOG_DIR");
  const std::string out_dir = (env_dir && *env_dir) ? std::string(env_dir) : options_.output_dir;

  std::ostringstream oss;
  oss << out_dir << "/ablation_drive_" << TimestampStringNow() << ".csv";
  return oss.str();
}

bool AlgorithmAblationLogger::Start(std::string* out_error) {
  const char* env_enable = std::getenv("ADAS_ABLATION_LOG_ENABLE");
  options_.enabled = ParseEnvBool(env_enable, options_.enabled);

  if (!options_.enabled) {
    running_ = false;
    return true;
  }

  if (!InitVirtualRoad(out_error)) {
    running_ = false;
    return false;
  }

  output_path_ = options_.output_path.empty() ? ResolveOutputPath() : options_.output_path;

  const std::string parent_dir = ParentDirOf(output_path_);
  if (!parent_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent_dir, ec);
    if (ec) {
      if (out_error) {
        *out_error = "failed to create output directory: " + parent_dir + " (" + ec.message() + ")";
      }
      return false;
    }
  }

  out_.open(output_path_, std::ios::out | std::ios::trunc);
  if (!out_.is_open()) {
    if (out_error) {
      *out_error = "failed to open output file: " + output_path_;
    }
    return false;
  }

  out_ << std::fixed << std::setprecision(6);
  WriteHeader();

  flush_counter_ = 0;
  route_vc_ = PoseState{};
  route_preview_mpc_ = PoseState{};
  route_disturbed_preview_mpc_ = PoseState{};
  route_raw_ = PoseState{};
  path_vc_.clear();
  path_preview_mpc_.clear();
  path_disturbed_preview_mpc_.clear();
  path_raw_.clear();
  path_vc_.emplace_back(0.0, 0.0);
  path_preview_mpc_.emplace_back(0.0, 0.0);
  path_disturbed_preview_mpc_.emplace_back(0.0, 0.0);
  path_raw_.emplace_back(0.0, 0.0);
  has_virtual_sim_config_ = false;
  summary_ = Summary{};

  start_steady_ = std::chrono::steady_clock::now();
  running_ = true;
  return true;
}

void AlgorithmAblationLogger::WriteHeader() {
  out_ << "unix_ms"
       << ",elapsed_s"
       << ",frame_idx"
       << ",frame_sync_ns"
       << ",dt_s"
       << ",ego_speed_kmh"
       << ",vc_speed_kmh"
       << ",vc_steer_deg"
       << ",vc_brake_0_10"
       << ",preview_mpc_speed_kmh"
       << ",preview_mpc_steer_deg"
       << ",preview_mpc_brake_0_10"
       << ",disturbed_preview_mpc_speed_kmh"
       << ",disturbed_preview_mpc_steer_base_deg"
       << ",disturbed_preview_mpc_steer_deg"
       << ",disturbed_preview_mpc_brake_0_10"
       << ",steering_disturbance_deg"
       << ",raw_speed_kmh"
       << ",raw_steer_deg"
       << ",raw_brake_0_10"
       << ",diff_speed_kmh"
       << ",diff_steer_deg"
       << ",diff_brake_0_10"
       << ",skeleton_vehicle_count"
       << ",skeleton_before_valid_count"
       << ",skeleton_after_valid_count"
       << ",skeleton_changed_count"
       << ",skeleton_heading_mean_abs_delta_deg"
       << ",skeleton_heading_max_abs_delta_deg"
       << ",route_vc_heading_rad"
       << ",route_vc_x_m"
       << ",route_vc_y_m"
       << ",route_vc_distance_m"
       << ",route_preview_mpc_heading_rad"
       << ",route_preview_mpc_x_m"
       << ",route_preview_mpc_y_m"
       << ",route_preview_mpc_distance_m"
       << ",route_disturbed_preview_mpc_heading_rad"
       << ",route_disturbed_preview_mpc_x_m"
       << ",route_disturbed_preview_mpc_y_m"
       << ",route_disturbed_preview_mpc_distance_m"
       << ",route_raw_heading_rad"
       << ",route_raw_x_m"
       << ",route_raw_y_m"
       << ",route_raw_distance_m"
       << ",route_xy_gap_m"
       << ",virtual_road_valid"
       << ",vc_cte_m"
       << ",vc_heading_err_deg"
       << ",vc_lane_departure"
       << ",preview_mpc_cte_m"
       << ",preview_mpc_heading_err_deg"
       << ",preview_mpc_lane_departure"
       << ",disturbed_preview_mpc_cte_m"
       << ",disturbed_preview_mpc_heading_err_deg"
       << ",disturbed_preview_mpc_lane_departure"
       << ",raw_cte_m"
       << ",raw_heading_err_deg"
       << ",raw_lane_departure"
       << '\n';
}

void AlgorithmAblationLogger::UpdatePose(PoseState* pose,
                                         double speed_kmh,
                                         double steer_deg,
                                         double dt_s) const {
  if (!pose) return;

  const double speed_mps = std::max(0.0, speed_kmh) / 3.6;
  const double steering_ratio = std::max(1e-3, options_.steering_ratio);
  const double wheelbase_m = std::max(1e-3, options_.wheelbase_m);

  const double road_wheel_deg = steer_deg / steering_ratio;
  const double delta_rad = DegToRad(road_wheel_deg);
  const double yaw_rate_rps = speed_mps * std::tan(delta_rad) / wheelbase_m;

  pose->heading_rad = WrapPi(pose->heading_rad + yaw_rate_rps * dt_s);

  const double ds = speed_mps * dt_s;
  pose->x_m += ds * std::cos(pose->heading_rad);
  pose->y_m += ds * std::sin(pose->heading_rad);
  pose->distance_m += ds;
}

AlgorithmAblationResult AlgorithmAblationLogger::Step(const AlgorithmAblationFrame& frame) {
  AlgorithmAblationResult result;
  if (!running_) return result;

  SkeletonDiff skeleton_diff;
  if (frame.world_before_skeleton && frame.world_after_skeleton) {
    std::unordered_map<int, const TrackingBox*> before_map;
    before_map.reserve(frame.world_before_skeleton->size());

    for (const auto& tb : *frame.world_before_skeleton) {
      if (tb.class_id != 1) continue;
      before_map[tb.id] = &tb;
      if (tb.target_heading_valid && IsFinite(tb.target_heading_deg)) {
        skeleton_diff.before_valid_count += 1;
      }
    }

    double heading_abs_delta_sum = 0.0;
    int heading_delta_count = 0;

    for (const auto& tb_after : *frame.world_after_skeleton) {
      if (tb_after.class_id != 1) continue;

      skeleton_diff.vehicle_count += 1;
      const bool after_valid = tb_after.target_heading_valid && IsFinite(tb_after.target_heading_deg);
      if (after_valid) {
        skeleton_diff.after_valid_count += 1;
      }

      const auto it = before_map.find(tb_after.id);
      if (it == before_map.end()) {
        if (after_valid) skeleton_diff.changed_count += 1;
        continue;
      }

      const auto& tb_before = *it->second;
      const bool before_valid = tb_before.target_heading_valid && IsFinite(tb_before.target_heading_deg);

      if (before_valid != after_valid) {
        skeleton_diff.changed_count += 1;
      }

      if (before_valid && after_valid) {
        const double delta_deg = NormalizeDeltaDeg(
            static_cast<double>(tb_after.target_heading_deg) -
            static_cast<double>(tb_before.target_heading_deg));
        const double abs_delta_deg = std::abs(delta_deg);

        heading_abs_delta_sum += abs_delta_deg;
        heading_delta_count += 1;
        skeleton_diff.max_abs_delta_deg = std::max(skeleton_diff.max_abs_delta_deg, abs_delta_deg);

        if (abs_delta_deg > 1e-3) {
          skeleton_diff.changed_count += 1;
        }
      }
    }

    if (heading_delta_count > 0) {
      skeleton_diff.mean_abs_delta_deg = heading_abs_delta_sum / static_cast<double>(heading_delta_count);
    }
  }

  result.skeleton_vehicle_count = skeleton_diff.vehicle_count;
  result.skeleton_before_valid_count = skeleton_diff.before_valid_count;
  result.skeleton_after_valid_count = skeleton_diff.after_valid_count;
  result.skeleton_changed_count = skeleton_diff.changed_count;
  result.skeleton_heading_mean_abs_delta_deg = skeleton_diff.mean_abs_delta_deg;
  result.skeleton_heading_max_abs_delta_deg = skeleton_diff.max_abs_delta_deg;

  const auto& vc_cmd = frame.vehicle_control_cmd;
  const double vc_speed_kmh = std::max(0.0, static_cast<double>(vc_cmd.speed_kmh));
  const double vc_steer_deg = static_cast<double>(vc_cmd.steer_deg);
  const double vc_brake_0_10 = Clamp(static_cast<double>(vc_cmd.brake_0_10), 0.0, 10.0);

  bool preview_mpc_active = frame.preview_mpc_valid;
  double preview_mpc_speed_kmh = std::numeric_limits<double>::quiet_NaN();
  double preview_mpc_steer_deg = std::numeric_limits<double>::quiet_NaN();
  double preview_mpc_brake_0_10 = std::numeric_limits<double>::quiet_NaN();
  if (preview_mpc_active) {
    preview_mpc_speed_kmh = std::max(0.0, frame.preview_mpc_speed_kmh);
    preview_mpc_steer_deg = frame.preview_mpc_steer_deg;
    preview_mpc_brake_0_10 = Clamp(frame.preview_mpc_brake_0_10, 0.0, 10.0);
    if (preview_mpc_brake_0_10 > 1e-3) {
      preview_mpc_speed_kmh = 0.0;
    }
  }

  bool disturbed_preview_mpc_active = frame.disturbed_preview_mpc_valid;
  double disturbed_preview_mpc_speed_kmh = std::numeric_limits<double>::quiet_NaN();
  double disturbed_preview_mpc_steer_base_deg = std::numeric_limits<double>::quiet_NaN();
  double disturbed_preview_mpc_steer_deg = std::numeric_limits<double>::quiet_NaN();
  double disturbed_preview_mpc_brake_0_10 = std::numeric_limits<double>::quiet_NaN();
  if (disturbed_preview_mpc_active) {
    disturbed_preview_mpc_speed_kmh = std::max(0.0, frame.disturbed_preview_mpc_speed_kmh);
    disturbed_preview_mpc_steer_base_deg = frame.disturbed_preview_mpc_steer_base_deg;
    disturbed_preview_mpc_steer_deg = frame.disturbed_preview_mpc_steer_deg;
    disturbed_preview_mpc_brake_0_10 = Clamp(frame.disturbed_preview_mpc_brake_0_10, 0.0, 10.0);
    if (disturbed_preview_mpc_brake_0_10 > 1e-3) {
      disturbed_preview_mpc_speed_kmh = 0.0;
    }
  }

  double raw_speed_kmh = std::max(0.0, static_cast<double>(vc_cmd.acc_cmd.speed_kmh));
  const double raw_steer_deg = static_cast<double>(vc_cmd.lka_steer_deg_raw);
  const double raw_brake_0_10 = Clamp(static_cast<double>(vc_cmd.acc_cmd.brake_0_10), 0.0, 10.0);
  if (raw_brake_0_10 > 1e-3) {
    raw_speed_kmh = 0.0;
  }

  result.vc_raw_speed_diff_kmh = vc_speed_kmh - raw_speed_kmh;
  result.vc_raw_steer_diff_deg = vc_steer_deg - raw_steer_deg;
  result.vc_raw_brake_diff_0_10 = vc_brake_0_10 - raw_brake_0_10;

  const double dt_s = Clamp(frame.dt_s, 0.001, 1.0);
  UpdatePose(&route_vc_, vc_speed_kmh, vc_steer_deg, dt_s);
  if (preview_mpc_active) {
    UpdatePose(&route_preview_mpc_, preview_mpc_speed_kmh, preview_mpc_steer_deg, dt_s);
  }
  if (disturbed_preview_mpc_active) {
    UpdatePose(&route_disturbed_preview_mpc_,
               disturbed_preview_mpc_speed_kmh,
               disturbed_preview_mpc_steer_deg,
               dt_s);
  }
  UpdatePose(&route_raw_, raw_speed_kmh, raw_steer_deg, dt_s);

  path_vc_.emplace_back(route_vc_.x_m, route_vc_.y_m);
  if (preview_mpc_active) {
    path_preview_mpc_.emplace_back(route_preview_mpc_.x_m, route_preview_mpc_.y_m);
  }
  if (disturbed_preview_mpc_active) {
    path_disturbed_preview_mpc_.emplace_back(route_disturbed_preview_mpc_.x_m,
                                             route_disturbed_preview_mpc_.y_m);
  }
  path_raw_.emplace_back(route_raw_.x_m, route_raw_.y_m);

  const double dx_gap_m = route_vc_.x_m - route_raw_.x_m;
  const double dy_gap_m = route_vc_.y_m - route_raw_.y_m;
  result.route_gap_m = std::sqrt(dx_gap_m * dx_gap_m + dy_gap_m * dy_gap_m);

  if (virtual_road_active_) {
    const VirtualRoadProjection vc_proj = ProjectPoseToPath(
        cv::Point2d(route_vc_.x_m, route_vc_.y_m),
        route_vc_.heading_rad,
        virtual_road_path_);
    const VirtualRoadProjection preview_mpc_proj = preview_mpc_active
        ? ProjectPoseToPath(
              cv::Point2d(route_preview_mpc_.x_m, route_preview_mpc_.y_m),
              route_preview_mpc_.heading_rad,
              virtual_road_path_)
        : VirtualRoadProjection{};
    const VirtualRoadProjection disturbed_preview_mpc_proj = disturbed_preview_mpc_active
        ? ProjectPoseToPath(
              cv::Point2d(route_disturbed_preview_mpc_.x_m, route_disturbed_preview_mpc_.y_m),
              route_disturbed_preview_mpc_.heading_rad,
              virtual_road_path_)
        : VirtualRoadProjection{};
    const VirtualRoadProjection raw_proj = ProjectPoseToPath(
        cv::Point2d(route_raw_.x_m, route_raw_.y_m),
        route_raw_.heading_rad,
        virtual_road_path_);

    if (vc_proj.valid && raw_proj.valid &&
        (!preview_mpc_active || preview_mpc_proj.valid) &&
        (!disturbed_preview_mpc_active || disturbed_preview_mpc_proj.valid)) {
      result.virtual_road_valid = true;
      result.vc_cte_m = vc_proj.cte_m;
      result.vc_heading_err_deg = vc_proj.heading_err_deg;
      if (preview_mpc_active) {
        result.preview_mpc_cte_m = preview_mpc_proj.cte_m;
        result.preview_mpc_heading_err_deg = preview_mpc_proj.heading_err_deg;
      }
      if (disturbed_preview_mpc_active) {
        result.disturbed_preview_mpc_cte_m = disturbed_preview_mpc_proj.cte_m;
        result.disturbed_preview_mpc_heading_err_deg = disturbed_preview_mpc_proj.heading_err_deg;
      }
      result.raw_cte_m = raw_proj.cte_m;
      result.raw_heading_err_deg = raw_proj.heading_err_deg;

      const double lane_half_width_m = options_.virtual_road_lane_width_m * 0.5;
      result.vc_lane_departure = (std::abs(result.vc_cte_m) > lane_half_width_m) ? 1 : 0;
      if (preview_mpc_active) {
        result.preview_mpc_lane_departure =
            (std::abs(result.preview_mpc_cte_m) > lane_half_width_m) ? 1 : 0;
      }
      if (disturbed_preview_mpc_active) {
        result.disturbed_preview_mpc_lane_departure =
            (std::abs(result.disturbed_preview_mpc_cte_m) > lane_half_width_m) ? 1 : 0;
      }
      result.raw_lane_departure = (std::abs(result.raw_cte_m) > lane_half_width_m) ? 1 : 0;
    }
  }

  summary_.sample_count += 1;
  if (result.skeleton_changed_count > 0) {
    summary_.skeleton_changed_frames += 1;
  }
  summary_.sum_skeleton_heading_mean_abs_delta_deg += result.skeleton_heading_mean_abs_delta_deg;
  summary_.max_skeleton_heading_abs_delta_deg = std::max(
      summary_.max_skeleton_heading_abs_delta_deg,
      result.skeleton_heading_max_abs_delta_deg);

  const double abs_speed_diff = std::abs(result.vc_raw_speed_diff_kmh);
  const double abs_steer_diff = std::abs(result.vc_raw_steer_diff_deg);
  const double abs_brake_diff = std::abs(result.vc_raw_brake_diff_0_10);

  summary_.sum_abs_speed_diff_kmh += abs_speed_diff;
  summary_.max_abs_speed_diff_kmh = std::max(summary_.max_abs_speed_diff_kmh, abs_speed_diff);
  summary_.sum_abs_steer_diff_deg += abs_steer_diff;
  summary_.max_abs_steer_diff_deg = std::max(summary_.max_abs_steer_diff_deg, abs_steer_diff);
  summary_.sum_abs_brake_diff_0_10 += abs_brake_diff;
  summary_.max_abs_brake_diff_0_10 = std::max(summary_.max_abs_brake_diff_0_10, abs_brake_diff);

  summary_.max_route_gap_m = std::max(summary_.max_route_gap_m, result.route_gap_m);
  summary_.final_route_gap_m = result.route_gap_m;

  if (result.virtual_road_valid) {
    summary_.virtual_road_valid_frames += 1;
    summary_.vc_lane_departure_count += static_cast<uint64_t>(result.vc_lane_departure);
    summary_.preview_mpc_lane_departure_count += static_cast<uint64_t>(result.preview_mpc_lane_departure);
    summary_.disturbed_preview_mpc_lane_departure_count +=
        static_cast<uint64_t>(result.disturbed_preview_mpc_lane_departure);
    summary_.raw_lane_departure_count += static_cast<uint64_t>(result.raw_lane_departure);

    const double abs_vc_cte = std::abs(result.vc_cte_m);
    const double abs_preview_mpc_cte = std::abs(result.preview_mpc_cte_m);
    const double abs_disturbed_preview_mpc_cte = std::abs(result.disturbed_preview_mpc_cte_m);
    const double abs_raw_cte = std::abs(result.raw_cte_m);
    const double abs_vc_heading = std::abs(result.vc_heading_err_deg);
    const double abs_preview_mpc_heading = std::abs(result.preview_mpc_heading_err_deg);
    const double abs_disturbed_preview_mpc_heading =
        std::abs(result.disturbed_preview_mpc_heading_err_deg);
    const double abs_raw_heading = std::abs(result.raw_heading_err_deg);

    summary_.sum_abs_vc_cte_m += abs_vc_cte;
    summary_.max_abs_vc_cte_m = std::max(summary_.max_abs_vc_cte_m, abs_vc_cte);
    summary_.sum_abs_preview_mpc_cte_m += abs_preview_mpc_cte;
    summary_.max_abs_preview_mpc_cte_m = std::max(summary_.max_abs_preview_mpc_cte_m, abs_preview_mpc_cte);
    summary_.sum_abs_disturbed_preview_mpc_cte_m += abs_disturbed_preview_mpc_cte;
    summary_.max_abs_disturbed_preview_mpc_cte_m =
        std::max(summary_.max_abs_disturbed_preview_mpc_cte_m, abs_disturbed_preview_mpc_cte);
    summary_.sum_abs_raw_cte_m += abs_raw_cte;
    summary_.max_abs_raw_cte_m = std::max(summary_.max_abs_raw_cte_m, abs_raw_cte);
    summary_.sum_abs_vc_heading_err_deg += abs_vc_heading;
    summary_.max_abs_vc_heading_err_deg = std::max(summary_.max_abs_vc_heading_err_deg, abs_vc_heading);
    summary_.sum_abs_preview_mpc_heading_err_deg += abs_preview_mpc_heading;
    summary_.max_abs_preview_mpc_heading_err_deg = std::max(
        summary_.max_abs_preview_mpc_heading_err_deg, abs_preview_mpc_heading);
    summary_.sum_abs_disturbed_preview_mpc_heading_err_deg += abs_disturbed_preview_mpc_heading;
    summary_.max_abs_disturbed_preview_mpc_heading_err_deg = std::max(
        summary_.max_abs_disturbed_preview_mpc_heading_err_deg,
        abs_disturbed_preview_mpc_heading);
    summary_.sum_abs_raw_heading_err_deg += abs_raw_heading;
    summary_.max_abs_raw_heading_err_deg = std::max(summary_.max_abs_raw_heading_err_deg, abs_raw_heading);
  }

  const auto now_wall = std::chrono::system_clock::now();
  const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now_wall.time_since_epoch()).count();
  const double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start_steady_).count();

  const double vc_cte = result.virtual_road_valid ? result.vc_cte_m : std::numeric_limits<double>::quiet_NaN();
  const double vc_heading = result.virtual_road_valid
      ? result.vc_heading_err_deg
      : std::numeric_limits<double>::quiet_NaN();
  const double preview_mpc_cte = result.virtual_road_valid
      ? result.preview_mpc_cte_m
      : std::numeric_limits<double>::quiet_NaN();
  const double preview_mpc_heading = result.virtual_road_valid
      ? result.preview_mpc_heading_err_deg
      : std::numeric_limits<double>::quiet_NaN();
  const double disturbed_preview_mpc_cte = result.virtual_road_valid
      ? result.disturbed_preview_mpc_cte_m
      : std::numeric_limits<double>::quiet_NaN();
  const double disturbed_preview_mpc_heading = result.virtual_road_valid
      ? result.disturbed_preview_mpc_heading_err_deg
      : std::numeric_limits<double>::quiet_NaN();
  const double route_preview_mpc_heading = preview_mpc_active
      ? route_preview_mpc_.heading_rad
      : std::numeric_limits<double>::quiet_NaN();
  const double route_preview_mpc_x = preview_mpc_active
      ? route_preview_mpc_.x_m
      : std::numeric_limits<double>::quiet_NaN();
  const double route_preview_mpc_y = preview_mpc_active
      ? route_preview_mpc_.y_m
      : std::numeric_limits<double>::quiet_NaN();
  const double route_preview_mpc_distance = preview_mpc_active
      ? route_preview_mpc_.distance_m
      : std::numeric_limits<double>::quiet_NaN();
  const double route_disturbed_preview_mpc_heading = disturbed_preview_mpc_active
      ? route_disturbed_preview_mpc_.heading_rad
      : std::numeric_limits<double>::quiet_NaN();
  const double route_disturbed_preview_mpc_x = disturbed_preview_mpc_active
      ? route_disturbed_preview_mpc_.x_m
      : std::numeric_limits<double>::quiet_NaN();
  const double route_disturbed_preview_mpc_y = disturbed_preview_mpc_active
      ? route_disturbed_preview_mpc_.y_m
      : std::numeric_limits<double>::quiet_NaN();
  const double route_disturbed_preview_mpc_distance = disturbed_preview_mpc_active
      ? route_disturbed_preview_mpc_.distance_m
      : std::numeric_limits<double>::quiet_NaN();
  const double raw_cte = result.virtual_road_valid ? result.raw_cte_m : std::numeric_limits<double>::quiet_NaN();
  const double raw_heading = result.virtual_road_valid
      ? result.raw_heading_err_deg
      : std::numeric_limits<double>::quiet_NaN();

  out_ << unix_ms
       << ',' << elapsed_s
       << ',' << frame.frame_index
       << ',' << frame.frame_sync_ns
       << ',' << FiniteOrNaN(dt_s)
       << ',' << FiniteOrNaN(frame.ego_speed_kmh)
       << ',' << FiniteOrNaN(vc_speed_kmh)
       << ',' << FiniteOrNaN(vc_steer_deg)
       << ',' << FiniteOrNaN(vc_brake_0_10)
       << ',' << FiniteOrNaN(preview_mpc_speed_kmh)
       << ',' << FiniteOrNaN(preview_mpc_steer_deg)
       << ',' << FiniteOrNaN(preview_mpc_brake_0_10)
       << ',' << FiniteOrNaN(disturbed_preview_mpc_speed_kmh)
       << ',' << FiniteOrNaN(disturbed_preview_mpc_steer_base_deg)
       << ',' << FiniteOrNaN(disturbed_preview_mpc_steer_deg)
       << ',' << FiniteOrNaN(disturbed_preview_mpc_brake_0_10)
       << ',' << FiniteOrNaN(frame.steering_disturbance_deg)
       << ',' << FiniteOrNaN(raw_speed_kmh)
       << ',' << FiniteOrNaN(raw_steer_deg)
       << ',' << FiniteOrNaN(raw_brake_0_10)
       << ',' << FiniteOrNaN(result.vc_raw_speed_diff_kmh)
       << ',' << FiniteOrNaN(result.vc_raw_steer_diff_deg)
       << ',' << FiniteOrNaN(result.vc_raw_brake_diff_0_10)
       << ',' << result.skeleton_vehicle_count
       << ',' << result.skeleton_before_valid_count
       << ',' << result.skeleton_after_valid_count
       << ',' << result.skeleton_changed_count
       << ',' << FiniteOrNaN(result.skeleton_heading_mean_abs_delta_deg)
       << ',' << FiniteOrNaN(result.skeleton_heading_max_abs_delta_deg)
       << ',' << FiniteOrNaN(route_vc_.heading_rad)
       << ',' << FiniteOrNaN(route_vc_.x_m)
       << ',' << FiniteOrNaN(route_vc_.y_m)
       << ',' << FiniteOrNaN(route_vc_.distance_m)
       << ',' << FiniteOrNaN(route_preview_mpc_heading)
       << ',' << FiniteOrNaN(route_preview_mpc_x)
       << ',' << FiniteOrNaN(route_preview_mpc_y)
       << ',' << FiniteOrNaN(route_preview_mpc_distance)
       << ',' << FiniteOrNaN(route_disturbed_preview_mpc_heading)
       << ',' << FiniteOrNaN(route_disturbed_preview_mpc_x)
       << ',' << FiniteOrNaN(route_disturbed_preview_mpc_y)
       << ',' << FiniteOrNaN(route_disturbed_preview_mpc_distance)
       << ',' << FiniteOrNaN(route_raw_.heading_rad)
       << ',' << FiniteOrNaN(route_raw_.x_m)
       << ',' << FiniteOrNaN(route_raw_.y_m)
       << ',' << FiniteOrNaN(route_raw_.distance_m)
       << ',' << FiniteOrNaN(result.route_gap_m)
       << ',' << (result.virtual_road_valid ? 1 : 0)
       << ',' << FiniteOrNaN(vc_cte)
       << ',' << FiniteOrNaN(vc_heading)
       << ',' << result.vc_lane_departure
       << ',' << FiniteOrNaN(preview_mpc_cte)
       << ',' << FiniteOrNaN(preview_mpc_heading)
       << ',' << result.preview_mpc_lane_departure
       << ',' << FiniteOrNaN(disturbed_preview_mpc_cte)
       << ',' << FiniteOrNaN(disturbed_preview_mpc_heading)
       << ',' << result.disturbed_preview_mpc_lane_departure
       << ',' << FiniteOrNaN(raw_cte)
       << ',' << FiniteOrNaN(raw_heading)
       << ',' << result.raw_lane_departure
       << '\n';

  flush_counter_ += 1;
  if (flush_counter_ >= static_cast<uint64_t>(std::max(1, options_.flush_every_n))) {
    out_.flush();
    flush_counter_ = 0;
  }

  return result;
}

bool AlgorithmAblationLogger::RunVirtualRoadSimulation(const VirtualRoadSimulationOptions& sim,
                                                       std::string* out_error) {
  if (!running_) {
    if (out_error) *out_error = "ablation logger is not running";
    return false;
  }
  if (!virtual_road_active_ || virtual_road_path_.size() < 2) {
    if (out_error) *out_error = "virtual road is not active; cannot run simulation";
    return false;
  }

  VirtualRoadSimulationOptions cfg = sim;
  cfg.frame_count = std::max<uint64_t>(1, cfg.frame_count);
  cfg.dt_s = Clamp(cfg.dt_s, 0.001, 1.0);
  cfg.speed_kmh = std::max(0.0, cfg.speed_kmh);
  cfg.max_steer_deg = std::max(1.0, cfg.max_steer_deg);
  cfg.raw_steer_osc_period_s = std::max(0.1, cfg.raw_steer_osc_period_s);
  last_virtual_sim_config_ = cfg;
  has_virtual_sim_config_ = true;

  // Reset trajectories/statistics to keep this run self-contained.
  route_vc_ = PoseState{};
  route_preview_mpc_ = PoseState{};
  route_disturbed_preview_mpc_ = PoseState{};
  route_raw_ = PoseState{};
  path_vc_.clear();
  path_preview_mpc_.clear();
  path_disturbed_preview_mpc_.clear();
  path_raw_.clear();
  summary_ = Summary{};
  flush_counter_ = 0;

  // Align initial pose with virtual road start tangent.
  const cv::Point2d p0 = virtual_road_path_[0];
  const cv::Point2d p1 = virtual_road_path_[1];
  const double init_heading_rad = std::atan2(p1.y - p0.y, p1.x - p0.x);
  route_vc_.x_m = p0.x;
  route_vc_.y_m = p0.y;
  route_vc_.heading_rad = init_heading_rad;
  route_preview_mpc_ = route_vc_;
  route_disturbed_preview_mpc_ = route_vc_;
  route_raw_ = route_vc_;
  path_vc_.emplace_back(route_vc_.x_m, route_vc_.y_m);
  path_preview_mpc_.emplace_back(route_preview_mpc_.x_m, route_preview_mpc_.y_m);
  path_disturbed_preview_mpc_.emplace_back(route_disturbed_preview_mpc_.x_m,
                                           route_disturbed_preview_mpc_.y_m);
  path_raw_.emplace_back(route_raw_.x_m, route_raw_.y_m);

  constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
  const double osc_w = kTwoPi / cfg.raw_steer_osc_period_s;
  double last_preview_mpc_steer_deg = 0.0;
  double last_disturbed_preview_mpc_steer_base_deg = 0.0;

  for (uint64_t i = 0; i < cfg.frame_count; ++i) {
    const double t_s = static_cast<double>(i) * cfg.dt_s;

    const VirtualRoadProjection vc_proj = ProjectPoseToPath(
        cv::Point2d(route_vc_.x_m, route_vc_.y_m),
        route_vc_.heading_rad,
        virtual_road_path_);
    const VirtualRoadProjection preview_mpc_proj = cfg.preview_mpc_enable
        ? ProjectPoseToPath(
              cv::Point2d(route_preview_mpc_.x_m, route_preview_mpc_.y_m),
              route_preview_mpc_.heading_rad,
              virtual_road_path_)
        : VirtualRoadProjection{};
    const VirtualRoadProjection disturbed_preview_mpc_proj =
        (cfg.preview_mpc_enable && cfg.disturbed_preview_mpc_enable)
            ? ProjectPoseToPath(
                  cv::Point2d(route_disturbed_preview_mpc_.x_m,
                              route_disturbed_preview_mpc_.y_m),
                  route_disturbed_preview_mpc_.heading_rad,
                  virtual_road_path_)
            : VirtualRoadProjection{};
    const VirtualRoadProjection raw_proj = ProjectPoseToPath(
        cv::Point2d(route_raw_.x_m, route_raw_.y_m),
        route_raw_.heading_rad,
        virtual_road_path_);

    const double steer_disturbance_deg =
        cfg.raw_steer_bias_deg + cfg.raw_steer_osc_amp_deg * std::sin(osc_w * t_s);

    double vc_steer_deg = 0.0;
    if (vc_proj.valid) {
      vc_steer_deg =
          -(cfg.vc_k_cte * vc_proj.cte_m + cfg.vc_k_heading * vc_proj.heading_err_deg);
    }
    vc_steer_deg = Clamp(vc_steer_deg, -cfg.max_steer_deg, cfg.max_steer_deg);

    double raw_steer_deg = 0.0;
    if (raw_proj.valid) {
      raw_steer_deg =
          -(cfg.raw_k_cte * raw_proj.cte_m + cfg.raw_k_heading * raw_proj.heading_err_deg);
    }
    raw_steer_deg += steer_disturbance_deg;
    raw_steer_deg = Clamp(raw_steer_deg, -cfg.max_steer_deg, cfg.max_steer_deg);

    double preview_mpc_steer_deg = 0.0;
    if (cfg.preview_mpc_enable && preview_mpc_proj.valid) {
      preview_mpc_steer_deg = ComputePreviewMpcSteerDeg(
          preview_mpc_proj,
          cfg.speed_kmh,
          cfg.dt_s,
          options_.wheelbase_m,
          options_.steering_ratio,
          cfg.max_steer_deg,
          last_preview_mpc_steer_deg,
          cfg);
      preview_mpc_steer_deg = Clamp(preview_mpc_steer_deg, -cfg.max_steer_deg, cfg.max_steer_deg);
      last_preview_mpc_steer_deg = preview_mpc_steer_deg;
    }

    double disturbed_preview_mpc_steer_base_deg = 0.0;
    double disturbed_preview_mpc_steer_deg = 0.0;
    const bool disturbed_preview_mpc_active =
        cfg.preview_mpc_enable && cfg.disturbed_preview_mpc_enable;
    if (disturbed_preview_mpc_active && disturbed_preview_mpc_proj.valid) {
      disturbed_preview_mpc_steer_base_deg = ComputePreviewMpcSteerDeg(
          disturbed_preview_mpc_proj,
          cfg.speed_kmh,
          cfg.dt_s,
          options_.wheelbase_m,
          options_.steering_ratio,
          cfg.max_steer_deg,
          last_disturbed_preview_mpc_steer_base_deg,
          cfg);
      disturbed_preview_mpc_steer_base_deg =
          Clamp(disturbed_preview_mpc_steer_base_deg, -cfg.max_steer_deg, cfg.max_steer_deg);
      last_disturbed_preview_mpc_steer_base_deg = disturbed_preview_mpc_steer_base_deg;
      disturbed_preview_mpc_steer_deg =
          Clamp(disturbed_preview_mpc_steer_base_deg + steer_disturbance_deg,
                -cfg.max_steer_deg,
                cfg.max_steer_deg);
    }

    AlgorithmAblationFrame frame;
    frame.frame_index = i;
    frame.frame_sync_ns = static_cast<uint64_t>(std::llround(t_s * 1e9));
    frame.dt_s = cfg.dt_s;
    frame.ego_speed_kmh = cfg.speed_kmh;
    frame.world_before_skeleton = nullptr;
    frame.world_after_skeleton = nullptr;
    frame.vehicle_control_cmd.steer_deg = static_cast<float>(vc_steer_deg);
    frame.vehicle_control_cmd.speed_kmh = static_cast<float>(cfg.speed_kmh);
    frame.vehicle_control_cmd.brake_0_10 = 0.0f;
    frame.vehicle_control_cmd.lka_steer_deg_raw = static_cast<float>(raw_steer_deg);
    frame.vehicle_control_cmd.acc_cmd.speed_kmh = static_cast<float>(cfg.speed_kmh);
    frame.vehicle_control_cmd.acc_cmd.brake_0_10 = 0.0f;
    frame.preview_mpc_valid = cfg.preview_mpc_enable;
    frame.preview_mpc_speed_kmh = cfg.speed_kmh;
    frame.preview_mpc_steer_deg = preview_mpc_steer_deg;
    frame.preview_mpc_brake_0_10 = 0.0;
    frame.disturbed_preview_mpc_valid = disturbed_preview_mpc_active;
    frame.disturbed_preview_mpc_speed_kmh = cfg.speed_kmh;
    frame.disturbed_preview_mpc_steer_base_deg = disturbed_preview_mpc_steer_base_deg;
    frame.disturbed_preview_mpc_steer_deg = disturbed_preview_mpc_steer_deg;
    frame.disturbed_preview_mpc_brake_0_10 = 0.0;
    frame.steering_disturbance_deg = steer_disturbance_deg;
    Step(frame);
  }

  if (out_.is_open()) out_.flush();
  return true;
}

void AlgorithmAblationLogger::WriteRoutePlot() {
  if (output_path_.empty()) return;
  if (path_vc_.empty() || path_raw_.empty()) return;

  const int plot_size_px = std::max(400, options_.plot_size_px);
  const int plot_margin_px = std::max(20, std::min(options_.plot_margin_px, plot_size_px / 3));

  cv::Mat canvas(plot_size_px, plot_size_px, CV_8UC3, cv::Scalar(250, 250, 250));

  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;

  auto expand_bounds = [&](const std::vector<cv::Point2d>& path) {
    for (const auto& p : path) {
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
    }
  };

  expand_bounds(path_vc_);
  expand_bounds(path_preview_mpc_);
  expand_bounds(path_disturbed_preview_mpc_);
  expand_bounds(path_raw_);
  if (virtual_road_active_) {
    expand_bounds(virtual_road_path_);
  }

  const double span_x = std::max(1.0, max_x - min_x);
  const double span_y = std::max(1.0, max_y - min_y);
  const double scale_x = (plot_size_px - 2.0 * plot_margin_px) / span_x;
  const double scale_y = (plot_size_px - 2.0 * plot_margin_px) / span_y;
  const double scale = std::max(1e-6, std::min(scale_x, scale_y));

  auto to_px = [&](const cv::Point2d& p) {
    const int x = static_cast<int>(std::llround(plot_margin_px + (p.x - min_x) * scale));
    const int y = static_cast<int>(std::llround(plot_size_px - plot_margin_px - (p.y - min_y) * scale));
    return cv::Point(x, y);
  };

  if (min_x <= 0.0 && max_x >= 0.0) {
    const cv::Point p0 = to_px(cv::Point2d(0.0, min_y));
    const cv::Point p1 = to_px(cv::Point2d(0.0, max_y));
    cv::line(canvas, p0, p1, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
  }
  if (min_y <= 0.0 && max_y >= 0.0) {
    const cv::Point p0 = to_px(cv::Point2d(min_x, 0.0));
    const cv::Point p1 = to_px(cv::Point2d(max_x, 0.0));
    cv::line(canvas, p0, p1, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
  }

  auto draw_path = [&](const std::vector<cv::Point2d>& path, const cv::Scalar& color, int thickness) {
    if (path.size() < 2) return;
    for (size_t i = 1; i < path.size(); ++i) {
      cv::line(canvas, to_px(path[i - 1]), to_px(path[i]), color, thickness, cv::LINE_AA);
    }
  };

  if (virtual_road_active_) {
    draw_path(virtual_road_path_, cv::Scalar(0, 180, 0), 2);
  }
  draw_path(path_vc_, cv::Scalar(255, 80, 0), 2);
  draw_path(path_preview_mpc_, cv::Scalar(160, 40, 180), 2);
  draw_path(path_disturbed_preview_mpc_, cv::Scalar(40, 40, 210), 2);
  draw_path(path_raw_, cv::Scalar(0, 170, 255), 2);

  const cv::Point start_pt = to_px(cv::Point2d(0.0, 0.0));
  cv::circle(canvas, start_pt, 5, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);

  const cv::Point vc_end = to_px(path_vc_.back());
  const cv::Point preview_mpc_end = to_px(path_preview_mpc_.back());
  const cv::Point disturbed_preview_mpc_end = to_px(path_disturbed_preview_mpc_.back());
  const cv::Point raw_end = to_px(path_raw_.back());
  cv::circle(canvas, vc_end, 4, cv::Scalar(255, 80, 0), cv::FILLED, cv::LINE_AA);
  cv::circle(canvas, preview_mpc_end, 4, cv::Scalar(160, 40, 180), cv::FILLED, cv::LINE_AA);
  cv::circle(canvas, disturbed_preview_mpc_end, 4, cv::Scalar(40, 40, 210), cv::FILLED, cv::LINE_AA);
  cv::circle(canvas, raw_end, 4, cv::Scalar(0, 170, 255), cv::FILLED, cv::LINE_AA);

  int legend_y = 40;
  if (virtual_road_active_) {
    cv::putText(canvas,
                "Reference road (" + virtual_road_mode_used_ + ")",
                cv::Point(30, legend_y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 180, 0),
                2,
                cv::LINE_AA);
    legend_y += 35;
  }

  cv::putText(canvas,
              "VC on",
              cv::Point(30, legend_y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(255, 80, 0),
              2,
              cv::LINE_AA);
  legend_y += 35;
  cv::putText(canvas,
              "Preview-MPC baseline",
              cv::Point(30, legend_y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(160, 40, 180),
              2,
              cv::LINE_AA);
  legend_y += 35;
  cv::putText(canvas,
              "Disturbed Preview-MPC comparator",
              cv::Point(30, legend_y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(40, 40, 210),
              2,
              cv::LINE_AA);
  legend_y += 35;
  cv::putText(canvas,
              "VC off (ACC+LKA raw)",
              cv::Point(30, legend_y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(0, 170, 255),
              2,
              cv::LINE_AA);
  legend_y += 35;

  std::ostringstream gap_oss;
  gap_oss << "Final route gap: " << std::fixed << std::setprecision(3) << summary_.final_route_gap_m << " m";
  cv::putText(canvas,
              gap_oss.str(),
              cv::Point(30, legend_y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.7,
              cv::Scalar(30, 30, 30),
              2,
              cv::LINE_AA);
  legend_y += 35;

  std::ostringstream axis_oss;
  axis_oss << "X span: " << std::fixed << std::setprecision(2) << span_x
           << " m, Y span: " << span_y << " m";
  cv::putText(canvas,
              axis_oss.str(),
              cv::Point(30, legend_y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.6,
              cv::Scalar(60, 60, 60),
              2,
              cv::LINE_AA);

  cv::imwrite(output_path_ + ".route.png", canvas);
}

void AlgorithmAblationLogger::WriteSummaryFile() {
  if (output_path_.empty()) return;

  std::ofstream summary(output_path_ + ".summary.txt", std::ios::out | std::ios::trunc);
  if (!summary.is_open()) return;

  summary << std::fixed << std::setprecision(6);
  summary << "output_csv=" << output_path_ << '\n';
  summary << "route_plot_png=" << output_path_ << ".route.png\n";
  summary << "samples=" << summary_.sample_count << '\n';
  summary << "route_vc_distance_m=" << route_vc_.distance_m << '\n';
  summary << "route_preview_mpc_distance_m=" << route_preview_mpc_.distance_m << '\n';
  summary << "route_disturbed_preview_mpc_distance_m="
          << route_disturbed_preview_mpc_.distance_m << '\n';
  summary << "route_raw_distance_m=" << route_raw_.distance_m << '\n';
  summary << "final_route_gap_m=" << summary_.final_route_gap_m << '\n';
  summary << "max_route_gap_m=" << summary_.max_route_gap_m << '\n';
  summary << "virtual_road_active=" << (virtual_road_active_ ? 1 : 0) << '\n';
  summary << "virtual_road_mode=" << (virtual_road_active_ ? virtual_road_mode_used_ : "disabled") << '\n';
  summary << "virtual_road_lane_width_m=" << options_.virtual_road_lane_width_m << '\n';
  if (has_virtual_sim_config_) {
    const auto& cfg = last_virtual_sim_config_;
    summary << "virtual_sim_frame_count=" << cfg.frame_count << '\n';
    summary << "virtual_sim_dt_s=" << cfg.dt_s << '\n';
    summary << "virtual_sim_speed_kmh=" << cfg.speed_kmh << '\n';
    summary << "virtual_sim_max_steer_deg=" << cfg.max_steer_deg << '\n';
    summary << "virtual_sim_vc_k_cte=" << cfg.vc_k_cte << '\n';
    summary << "virtual_sim_vc_k_heading=" << cfg.vc_k_heading << '\n';
    summary << "virtual_sim_raw_k_cte=" << cfg.raw_k_cte << '\n';
    summary << "virtual_sim_raw_k_heading=" << cfg.raw_k_heading << '\n';
    summary << "virtual_sim_raw_steer_bias_deg=" << cfg.raw_steer_bias_deg << '\n';
    summary << "virtual_sim_raw_steer_osc_amp_deg=" << cfg.raw_steer_osc_amp_deg << '\n';
    summary << "virtual_sim_raw_steer_osc_period_s=" << cfg.raw_steer_osc_period_s << '\n';
    summary << "preview_mpc_enable=" << (cfg.preview_mpc_enable ? 1 : 0) << '\n';
    summary << "disturbed_preview_mpc_enable="
            << (cfg.disturbed_preview_mpc_enable ? 1 : 0) << '\n';
    summary << "disturbed_preview_mpc_definition="
            << "preview_mpc_steer_base_deg plus the same steering_disturbance_deg used by raw baseline\n";
    summary << "preview_mpc_horizon=" << cfg.preview_mpc_horizon << '\n';
    summary << "preview_mpc_q_cte=" << cfg.preview_mpc_q_cte << '\n';
    summary << "preview_mpc_q_heading=" << cfg.preview_mpc_q_heading << '\n';
    summary << "preview_mpc_q_steer=" << cfg.preview_mpc_q_steer << '\n';
    summary << "preview_mpc_r_steer_rate=" << cfg.preview_mpc_r_steer_rate << '\n';
    summary << "preview_mpc_max_steer_rate_deg_s="
            << cfg.preview_mpc_max_steer_rate_deg_s << '\n';
  }

  if (summary_.sample_count > 0) {
    const double inv_n = 1.0 / static_cast<double>(summary_.sample_count);
    summary << "skeleton_changed_ratio="
            << static_cast<double>(summary_.skeleton_changed_frames) * inv_n << '\n';
    summary << "avg_skeleton_heading_mean_abs_delta_deg="
            << summary_.sum_skeleton_heading_mean_abs_delta_deg * inv_n << '\n';
    summary << "max_skeleton_heading_abs_delta_deg="
            << summary_.max_skeleton_heading_abs_delta_deg << '\n';

    summary << "avg_abs_vc_raw_speed_diff_kmh="
            << summary_.sum_abs_speed_diff_kmh * inv_n << '\n';
    summary << "max_abs_vc_raw_speed_diff_kmh="
            << summary_.max_abs_speed_diff_kmh << '\n';

    summary << "avg_abs_vc_raw_steer_diff_deg="
            << summary_.sum_abs_steer_diff_deg * inv_n << '\n';
    summary << "max_abs_vc_raw_steer_diff_deg="
            << summary_.max_abs_steer_diff_deg << '\n';

    summary << "avg_abs_vc_raw_brake_diff_0_10="
            << summary_.sum_abs_brake_diff_0_10 * inv_n << '\n';
    summary << "max_abs_vc_raw_brake_diff_0_10="
            << summary_.max_abs_brake_diff_0_10 << '\n';
  } else {
    summary << "skeleton_changed_ratio=0\n";
    summary << "avg_skeleton_heading_mean_abs_delta_deg=0\n";
    summary << "max_skeleton_heading_abs_delta_deg=0\n";
    summary << "avg_abs_vc_raw_speed_diff_kmh=0\n";
    summary << "max_abs_vc_raw_speed_diff_kmh=0\n";
    summary << "avg_abs_vc_raw_steer_diff_deg=0\n";
    summary << "max_abs_vc_raw_steer_diff_deg=0\n";
    summary << "avg_abs_vc_raw_brake_diff_0_10=0\n";
    summary << "max_abs_vc_raw_brake_diff_0_10=0\n";
  }

  if (summary_.virtual_road_valid_frames > 0) {
    const double inv_valid = 1.0 / static_cast<double>(summary_.virtual_road_valid_frames);
    summary << "virtual_road_valid_ratio="
            << static_cast<double>(summary_.virtual_road_valid_frames) /
                   std::max(1.0, static_cast<double>(summary_.sample_count))
            << '\n';
    summary << "avg_abs_vc_cte_m=" << summary_.sum_abs_vc_cte_m * inv_valid << '\n';
    summary << "max_abs_vc_cte_m=" << summary_.max_abs_vc_cte_m << '\n';
    summary << "avg_abs_preview_mpc_cte_m=" << summary_.sum_abs_preview_mpc_cte_m * inv_valid << '\n';
    summary << "max_abs_preview_mpc_cte_m=" << summary_.max_abs_preview_mpc_cte_m << '\n';
    summary << "avg_abs_disturbed_preview_mpc_cte_m="
            << summary_.sum_abs_disturbed_preview_mpc_cte_m * inv_valid << '\n';
    summary << "max_abs_disturbed_preview_mpc_cte_m="
            << summary_.max_abs_disturbed_preview_mpc_cte_m << '\n';
    summary << "avg_abs_raw_cte_m=" << summary_.sum_abs_raw_cte_m * inv_valid << '\n';
    summary << "max_abs_raw_cte_m=" << summary_.max_abs_raw_cte_m << '\n';
    summary << "avg_abs_vc_heading_err_deg=" << summary_.sum_abs_vc_heading_err_deg * inv_valid << '\n';
    summary << "max_abs_vc_heading_err_deg=" << summary_.max_abs_vc_heading_err_deg << '\n';
    summary << "avg_abs_preview_mpc_heading_err_deg="
            << summary_.sum_abs_preview_mpc_heading_err_deg * inv_valid << '\n';
    summary << "max_abs_preview_mpc_heading_err_deg="
            << summary_.max_abs_preview_mpc_heading_err_deg << '\n';
    summary << "avg_abs_disturbed_preview_mpc_heading_err_deg="
            << summary_.sum_abs_disturbed_preview_mpc_heading_err_deg * inv_valid << '\n';
    summary << "max_abs_disturbed_preview_mpc_heading_err_deg="
            << summary_.max_abs_disturbed_preview_mpc_heading_err_deg << '\n';
    summary << "avg_abs_raw_heading_err_deg=" << summary_.sum_abs_raw_heading_err_deg * inv_valid << '\n';
    summary << "max_abs_raw_heading_err_deg=" << summary_.max_abs_raw_heading_err_deg << '\n';
    summary << "vc_lane_departure_ratio="
            << static_cast<double>(summary_.vc_lane_departure_count) * inv_valid << '\n';
    summary << "preview_mpc_lane_departure_ratio="
            << static_cast<double>(summary_.preview_mpc_lane_departure_count) * inv_valid << '\n';
    summary << "disturbed_preview_mpc_lane_departure_ratio="
            << static_cast<double>(summary_.disturbed_preview_mpc_lane_departure_count) * inv_valid << '\n';
    summary << "raw_lane_departure_ratio="
            << static_cast<double>(summary_.raw_lane_departure_count) * inv_valid << '\n';
  } else {
    summary << "virtual_road_valid_ratio=0\n";
    summary << "avg_abs_vc_cte_m=nan\n";
    summary << "max_abs_vc_cte_m=nan\n";
    summary << "avg_abs_preview_mpc_cte_m=nan\n";
    summary << "max_abs_preview_mpc_cte_m=nan\n";
    summary << "avg_abs_disturbed_preview_mpc_cte_m=nan\n";
    summary << "max_abs_disturbed_preview_mpc_cte_m=nan\n";
    summary << "avg_abs_raw_cte_m=nan\n";
    summary << "max_abs_raw_cte_m=nan\n";
    summary << "avg_abs_vc_heading_err_deg=nan\n";
    summary << "max_abs_vc_heading_err_deg=nan\n";
    summary << "avg_abs_preview_mpc_heading_err_deg=nan\n";
    summary << "max_abs_preview_mpc_heading_err_deg=nan\n";
    summary << "avg_abs_disturbed_preview_mpc_heading_err_deg=nan\n";
    summary << "max_abs_disturbed_preview_mpc_heading_err_deg=nan\n";
    summary << "avg_abs_raw_heading_err_deg=nan\n";
    summary << "max_abs_raw_heading_err_deg=nan\n";
    summary << "vc_lane_departure_ratio=nan\n";
    summary << "preview_mpc_lane_departure_ratio=nan\n";
    summary << "disturbed_preview_mpc_lane_departure_ratio=nan\n";
    summary << "raw_lane_departure_ratio=nan\n";
  }
}

void AlgorithmAblationLogger::Stop() {
  if (!running_) return;

  WriteRoutePlot();
  WriteSummaryFile();

  if (out_.is_open()) {
    out_.flush();
    out_.close();
  }

  running_ = false;
}

}  // namespace ablation
