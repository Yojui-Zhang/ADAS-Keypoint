#include "skeleton_layout_resolver.h"

namespace adas_app {

vehicle_skeleton::SkeletonKptLayout ResolveSkeletonLayout(const AdasSystemConfig& runtime_cfg) {
  if (runtime_cfg.behavior.use_custom_layout) {
    return vehicle_skeleton::SkeletonKptLayout::FromIndexArray(runtime_cfg.behavior.custom_layout);
  }

#ifdef USE_TFLITE
  return vehicle_skeleton::SkeletonKptLayout::Default0123_4567_891011();
#else
  return vehicle_skeleton::SkeletonKptLayout::Default3456_78910_12131415();
#endif
}

}  // namespace adas_app
