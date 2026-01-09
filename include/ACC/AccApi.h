#pragma once
#include "AccConfig.h"
#include "AccController.h"
#include "config.h"
#include <vector>

// 你的 TrackingBox 在專案中已存在：只要 include 你自己的定義即可
// #include "TrackingBox.h"

namespace acc {

// 對外：只吃 WorldResult，吐 speed(km/h) + brake(0~10)
AccCommand ACC_Run(const std::vector<TrackingBox>& world_result);

// 可選：餵 CAN 車速提升精準度（但不破壞你要求的主函式介面）
void ACC_SetEgoSpeedKmh(float ego_speed_kmh);

// 可選：調參
void ACC_SetConfig(const AccConfig& cfg);
AccConfig ACC_GetConfig();

} // namespace acc

