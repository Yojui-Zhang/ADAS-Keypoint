#include "longitudinal_vehicle_state.h"

#include <atomic>
#include <cmath>

namespace controller {
namespace {

struct AtomicVehicleSpeedState {
  std::atomic<std::uint64_t> sequence{0};
  std::atomic<float> speed_kmh{0.0f};
  std::atomic<std::uint64_t> timestamp_ns{0};
};

struct AtomicVehicleAccelerationState {
  std::atomic<std::uint64_t> sequence{0};
  std::atomic<float> acceleration_mps2{0.0f};
  std::atomic<std::uint64_t> timestamp_ns{0};
};

AtomicVehicleSpeedState& GetSpeedState() noexcept {
  static AtomicVehicleSpeedState state;
  return state;
}

AtomicVehicleAccelerationState& GetAccelerationState() noexcept {
  static AtomicVehicleAccelerationState state;
  return state;
}

}  // namespace

void PublishLongitudinalVehicleSpeed(const float speed_kmh,
                                     const std::uint64_t timestamp_ns) noexcept {
  AtomicVehicleSpeedState& state = GetSpeedState();

  if (!std::isfinite(speed_kmh) || speed_kmh < 0.0f || timestamp_ns == 0) {
    return;
  }

  state.sequence.fetch_add(1, std::memory_order_acq_rel);
  state.speed_kmh.store(speed_kmh, std::memory_order_relaxed);
  state.timestamp_ns.store(timestamp_ns, std::memory_order_relaxed);
  state.sequence.fetch_add(1, std::memory_order_release);
}

void PublishLongitudinalVehicleAcceleration(
    const float acceleration_mps2,
    const std::uint64_t timestamp_ns) noexcept {
  AtomicVehicleAccelerationState& state = GetAccelerationState();

  if (!std::isfinite(acceleration_mps2) || timestamp_ns == 0) {
    return;
  }

  state.sequence.fetch_add(1, std::memory_order_acq_rel);
  state.acceleration_mps2.store(acceleration_mps2, std::memory_order_relaxed);
  state.timestamp_ns.store(timestamp_ns, std::memory_order_relaxed);
  state.sequence.fetch_add(1, std::memory_order_release);
}

LongitudinalVehicleSpeedSnapshot ReadLongitudinalVehicleSpeed() noexcept {
  AtomicVehicleSpeedState& state = GetSpeedState();

  for (int attempt = 0; attempt < 8; ++attempt) {
    const std::uint64_t sequence_begin =
        state.sequence.load(std::memory_order_acquire);
    if ((sequence_begin & 1U) != 0U) {
      continue;
    }

    LongitudinalVehicleSpeedSnapshot snapshot;
    snapshot.speed_kmh = state.speed_kmh.load(std::memory_order_relaxed);
    snapshot.timestamp_ns = state.timestamp_ns.load(std::memory_order_relaxed);

    const std::uint64_t sequence_end =
        state.sequence.load(std::memory_order_acquire);
    if (sequence_begin != sequence_end) {
      continue;
    }

    snapshot.valid = snapshot.timestamp_ns != 0 &&
                     std::isfinite(snapshot.speed_kmh) &&
                     snapshot.speed_kmh >= 0.0f;
    return snapshot;
  }

  return {};
}

LongitudinalVehicleAccelerationSnapshot
ReadLongitudinalVehicleAcceleration() noexcept {
  AtomicVehicleAccelerationState& state = GetAccelerationState();

  for (int attempt = 0; attempt < 8; ++attempt) {
    const std::uint64_t sequence_begin =
        state.sequence.load(std::memory_order_acquire);
    if ((sequence_begin & 1U) != 0U) {
      continue;
    }

    LongitudinalVehicleAccelerationSnapshot snapshot;
    snapshot.acceleration_mps2 =
        state.acceleration_mps2.load(std::memory_order_relaxed);
    snapshot.timestamp_ns = state.timestamp_ns.load(std::memory_order_relaxed);

    const std::uint64_t sequence_end =
        state.sequence.load(std::memory_order_acquire);
    if (sequence_begin != sequence_end) {
      continue;
    }

    snapshot.valid = snapshot.timestamp_ns != 0 &&
                     std::isfinite(snapshot.acceleration_mps2);
    return snapshot;
  }

  return {};
}

bool IsLongitudinalVehicleSpeedFresh(
    const LongitudinalVehicleSpeedSnapshot& snapshot,
    const std::uint64_t now_ns,
    const std::uint64_t timeout_ns) noexcept {
  if (!snapshot.valid || now_ns < snapshot.timestamp_ns) {
    return false;
  }

  return now_ns - snapshot.timestamp_ns <= timeout_ns;
}

bool IsLongitudinalVehicleAccelerationFresh(
    const LongitudinalVehicleAccelerationSnapshot& snapshot,
    const std::uint64_t now_ns,
    const std::uint64_t timeout_ns) noexcept {
  if (!snapshot.valid || now_ns < snapshot.timestamp_ns) {
    return false;
  }

  return now_ns - snapshot.timestamp_ns <= timeout_ns;
}

}  // namespace controller
