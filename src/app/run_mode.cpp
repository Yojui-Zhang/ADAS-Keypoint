#include "run_mode.h"

#include <algorithm>
#include <cctype>

namespace {

std::string ToLowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

}  // namespace

namespace adas_app {

RunMode ParseRunMode(const std::string& s) {
  const std::string mode = ToLowerCopy(s);
  if (mode == "virtual_road" || mode == "virtual-road" || mode == "virtual") {
    return RunMode::VirtualRoad;
  }
  if (mode == "real_car" || mode == "real-car" || mode == "real") {
    return RunMode::RealCar;
  }
  return RunMode::Video;
}

const char* RunModeName(RunMode mode) {
  switch (mode) {
    case RunMode::VirtualRoad: return "virtual_road";
    case RunMode::RealCar: return "real_car";
    default: return "video";
  }
}

}  // namespace adas_app
