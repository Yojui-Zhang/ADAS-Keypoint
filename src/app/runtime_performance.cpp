#include "runtime_performance.h"

namespace adas_app {

double ElapsedMs(const PerfClock::time_point& start,
                 const PerfClock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace adas_app
