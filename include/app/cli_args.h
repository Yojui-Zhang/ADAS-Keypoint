#pragma once

#include <iosfwd>
#include <string>

namespace adas_app {

struct CliArgs {
  const char* lanepose_model_path = nullptr;
  char* classify_model_path = nullptr;
  std::string system_config_path;
};

bool ParseCliArgs(int argc, char** argv, CliArgs& out_args);
void PrintCliUsage(std::ostream& os, const char* program_name);

}  // namespace adas_app
