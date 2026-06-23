#include "cli_args.h"

#include <ostream>

namespace adas_app {

bool ParseCliArgs(int argc, char** argv, CliArgs& out_args) {
  if (argc < 3) return false;

  out_args.lanepose_model_path = argv[1];
  out_args.classify_model_path = argv[2];

  if (argc >= 4) {
    out_args.system_config_path = argv[3];
  }
  return true;
}

void PrintCliUsage(std::ostream& os, const char* program_name) {
  os << "Usage: " << program_name
     << " <LanePose_Model_Path> <Classify_Model_Path> [System_Config_Path]";
}

}  // namespace adas_app
