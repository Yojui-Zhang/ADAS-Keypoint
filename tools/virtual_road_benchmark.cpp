#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "algorithm_ablation_logger.h"

namespace {

double ParseDouble(const std::string& label, const std::string& value) {
  char* endptr = nullptr;
  const double parsed = std::strtod(value.c_str(), &endptr);
  if (endptr == value.c_str() || (endptr && *endptr != '\0')) {
    throw std::runtime_error("invalid numeric value for " + label + ": " + value);
  }
  return parsed;
}

uint64_t ParseUInt64(const std::string& label, const std::string& value) {
  char* endptr = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &endptr, 10);
  if (endptr == value.c_str() || (endptr && *endptr != '\0')) {
    throw std::runtime_error("invalid integer value for " + label + ": " + value);
  }
  return static_cast<uint64_t>(parsed);
}

bool ParseBool(const std::string& label, const std::string& value) {
  if (value == "1" || value == "true" || value == "TRUE" ||
      value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" ||
      value == "no" || value == "off") {
    return false;
  }
  throw std::runtime_error("invalid boolean value for " + label + ": " + value);
}

void PrintUsage() {
  std::cerr
      << "Usage: virtual_road_benchmark --output <csv> [options]\n"
      << "Options:\n"
      << "  --speed-kmh <float>\n"
      << "  --frames <int>\n"
      << "  --dt-s <float>\n"
      << "  --max-steer-deg <float>\n"
      << "  --road-mode <straight|arc|s_curve|csv>\n"
      << "  --road-csv <path>\n"
      << "  --road-length-m <float>\n"
      << "  --road-step-m <float>\n"
      << "  --lane-width-m <float>\n"
      << "  --arc-radius-m <float>\n"
      << "  --s-amplitude-m <float>\n"
      << "  --s-wavelength-m <float>\n"
      << "  --steering-ratio <float>\n"
      << "  --wheelbase-m <float>\n"
      << "  --preview-mpc <0|1>\n"
      << "  --disturbed-preview-mpc <0|1>\n"
      << "  --raw-steer-bias-deg <float>\n"
      << "  --raw-steer-osc-amp-deg <float>\n"
      << "  --raw-steer-osc-period-s <float>\n";
}

}  // namespace

int main(int argc, char** argv) {
  ablation::AlgorithmAblationOptions options;
  ablation::VirtualRoadSimulationOptions sim;

  options.enabled = true;
  options.virtual_road_enable = true;
  options.virtual_road_mode = "csv";
  options.virtual_road_csv_path = "./ADAS/road_csv/s_curve.csv";
  options.virtual_road_lane_width_m = 3.76;

  std::string output_path;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto require_value = [&](const std::string& label) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for " + label);
        }
        return std::string(argv[++i]);
      };

      if (arg == "--output") {
        output_path = require_value(arg);
      } else if (arg == "--speed-kmh") {
        sim.speed_kmh = ParseDouble(arg, require_value(arg));
      } else if (arg == "--frames") {
        sim.frame_count = ParseUInt64(arg, require_value(arg));
      } else if (arg == "--dt-s") {
        sim.dt_s = ParseDouble(arg, require_value(arg));
      } else if (arg == "--max-steer-deg") {
        sim.max_steer_deg = ParseDouble(arg, require_value(arg));
      } else if (arg == "--road-mode") {
        options.virtual_road_mode = require_value(arg);
      } else if (arg == "--road-csv") {
        options.virtual_road_csv_path = require_value(arg);
      } else if (arg == "--road-length-m") {
        options.virtual_road_length_m = ParseDouble(arg, require_value(arg));
      } else if (arg == "--road-step-m") {
        options.virtual_road_step_m = ParseDouble(arg, require_value(arg));
      } else if (arg == "--lane-width-m") {
        options.virtual_road_lane_width_m = ParseDouble(arg, require_value(arg));
      } else if (arg == "--arc-radius-m") {
        options.virtual_road_arc_radius_m = ParseDouble(arg, require_value(arg));
      } else if (arg == "--s-amplitude-m") {
        options.virtual_road_s_amplitude_m = ParseDouble(arg, require_value(arg));
      } else if (arg == "--s-wavelength-m") {
        options.virtual_road_s_wavelength_m = ParseDouble(arg, require_value(arg));
      } else if (arg == "--steering-ratio") {
        options.steering_ratio = ParseDouble(arg, require_value(arg));
      } else if (arg == "--wheelbase-m") {
        options.wheelbase_m = ParseDouble(arg, require_value(arg));
      } else if (arg == "--preview-mpc") {
        sim.preview_mpc_enable = ParseBool(arg, require_value(arg));
      } else if (arg == "--disturbed-preview-mpc") {
        sim.disturbed_preview_mpc_enable = ParseBool(arg, require_value(arg));
      } else if (arg == "--raw-steer-bias-deg") {
        sim.raw_steer_bias_deg = ParseDouble(arg, require_value(arg));
      } else if (arg == "--raw-steer-osc-amp-deg") {
        sim.raw_steer_osc_amp_deg = ParseDouble(arg, require_value(arg));
      } else if (arg == "--raw-steer-osc-period-s") {
        sim.raw_steer_osc_period_s = ParseDouble(arg, require_value(arg));
      } else if (arg == "--help" || arg == "-h") {
        PrintUsage();
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }
  } catch (const std::exception& exc) {
    PrintUsage();
    std::cerr << "virtual_road_benchmark: " << exc.what() << '\n';
    return 2;
  }

  if (output_path.empty()) {
    PrintUsage();
    std::cerr << "virtual_road_benchmark: --output is required\n";
    return 2;
  }

  options.output_path = output_path;

  ablation::AlgorithmAblationLogger logger(options);
  std::string error;
  if (!logger.Start(&error)) {
    std::cerr << "virtual_road_benchmark: start failed: " << error << '\n';
    return 1;
  }
  if (!logger.RunVirtualRoadSimulation(sim, &error)) {
    std::cerr << "virtual_road_benchmark: simulation failed: " << error << '\n';
    logger.Stop();
    return 1;
  }
  logger.Stop();

  std::cout << "virtual_road_benchmark: wrote " << output_path << '\n';
  return 0;
}
