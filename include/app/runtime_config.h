#pragma once

#include <string>

#include "config.h"
#include "system_config.h"

namespace adas_app {

bool LoadRuntimeConfigWithFallback(const std::string& requested_path,
                                   AdasSystemConfig& out_cfg,
                                   std::string& out_loaded_path,
                                   std::string& out_error);

void ApplyTensorRtRuntimeConfig(const TensorRtRuntimeConfig& runtime_cfg,
                                Config& trt_cfg);

void ApplySubsystemConfig(const AdasSystemConfig& runtime_cfg);

}  // namespace adas_app
