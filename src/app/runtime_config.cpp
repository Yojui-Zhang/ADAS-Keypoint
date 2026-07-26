#include "runtime_config.h"

#include <vector>

#include "AccApi.h"
#include "GeometryFunction.h"
#include "VehicleControlApi.h"
#include "lane_keeping.h"
#include "throttle_control.h"

namespace adas_app {

bool LoadRuntimeConfigWithFallback(const std::string& requested_path,
                                   AdasSystemConfig& out_cfg,
                                   std::string& out_loaded_path,
                                   std::string& out_error) {
  std::vector<std::string> candidates;
  if (requested_path.empty() == false) {
    candidates.push_back(requested_path);
  } else {
    candidates.push_back("../config/system_config.yaml");
    candidates.push_back("./config/system_config.yaml");
    candidates.push_back("config/system_config.yaml");
  }

  for (const auto& path : candidates) {
    std::string err;
    if (LoadSystemConfig(path, out_cfg, &err)) {
      out_loaded_path = path;
      return true;
    }
    out_error = err;
  }

  return false;
}

void ApplyTensorRtRuntimeConfig(const TensorRtRuntimeConfig& runtime_cfg,
                                Config& trt_cfg) {
  trt_cfg.topk = runtime_cfg.topk;
  trt_cfg.score_thres = runtime_cfg.score_thres;
  trt_cfg.iou_thres = runtime_cfg.iou_thres;
  trt_cfg.num_labels = runtime_cfg.num_labels;
}

void ApplySubsystemConfig(const AdasSystemConfig& runtime_cfg) {
  Geometry_SetConfig(runtime_cfg.geometry);
  lane_keeping_set_control_config(runtime_cfg.lka);
  lane_keeping_reset_state();
  acc::ACC_SetConfig(runtime_cfg.acc);
  controller::ConfigureThrottleRuntime(runtime_cfg.throttle);
  stability::VehicleControl_SetStabilityConfig(runtime_cfg.stability);
}

}  // namespace adas_app
