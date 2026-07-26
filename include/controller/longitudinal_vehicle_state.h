#pragma once

#include <cstdint>

namespace controller {

struct LongitudinalVehicleSpeedSnapshot {
  float speed_kmh = 0.0f;
  std::uint64_t timestamp_ns = 0;
  bool valid = false;
};

struct LongitudinalVehicleAccelerationSnapshot {
  float acceleration_mps2 = 0.0f;
  std::uint64_t timestamp_ns = 0;
  bool valid = false;
};

void PublishLongitudinalVehicleSpeed(float speed_kmh,
                                     std::uint64_t timestamp_ns) noexcept;

void PublishLongitudinalVehicleAcceleration(
    float acceleration_mps2,
    std::uint64_t timestamp_ns) noexcept;

[[nodiscard]] LongitudinalVehicleSpeedSnapshot
ReadLongitudinalVehicleSpeed() noexcept;

[[nodiscard]] LongitudinalVehicleAccelerationSnapshot
ReadLongitudinalVehicleAcceleration() noexcept;

[[nodiscard]] bool IsLongitudinalVehicleSpeedFresh(
    const LongitudinalVehicleSpeedSnapshot& snapshot,
    std::uint64_t now_ns,
    std::uint64_t timeout_ns) noexcept;

[[nodiscard]] bool IsLongitudinalVehicleAccelerationFresh(
    const LongitudinalVehicleAccelerationSnapshot& snapshot,
    std::uint64_t now_ns,
    std::uint64_t timeout_ns) noexcept;

}  // namespace controller
