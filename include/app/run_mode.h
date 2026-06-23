#pragma once

#include <string>

namespace adas_app {

enum class RunMode {
  Video,
  VirtualRoad,
  RealCar
};

RunMode ParseRunMode(const std::string& s);
const char* RunModeName(RunMode mode);

}  // namespace adas_app
