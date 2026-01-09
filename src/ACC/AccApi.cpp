#include "AccApi.h"
#include "AccController.h"
#include <mutex>

namespace acc {

static std::mutex g_mtx;
static AccConfig g_cfg;
static AccController g_controller(g_cfg);

AccCommand ACC_Run(const std::vector<TrackingBox>& world_result) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_controller.SetConfig(g_cfg);
  return g_controller.Update(world_result);
}

void ACC_SetEgoSpeedKmh(float ego_speed_kmh) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_controller.SetEgoSpeedKmh(ego_speed_kmh);
}

void ACC_SetConfig(const AccConfig& cfg) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_cfg = cfg;
  g_controller.SetConfig(g_cfg);
}

AccConfig ACC_GetConfig() {
  std::lock_guard<std::mutex> lk(g_mtx);
  return g_cfg;
}

} // namespace acc

