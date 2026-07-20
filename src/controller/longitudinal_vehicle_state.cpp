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

AtomicVehicleSpeedState& GetState() noexcept {
  static AtomicVehicleSpeedState state;
  return state;
}

}  // namespace

void PublishLongitudinalVehicleSpeed(const float speed_kmh,
                                     const std::uint64_t timestamp_ns) noexcept {
  AtomicVehicleSpeedState& state = GetState();

  if (!std::isfinite(speed_kmh) || speed_kmh < 0.0f || timestamp_ns == 0) {
    return;
  }

  state.sequence.fetch_add(1, std::memory_order_acq_rel);
  state.speed_kmh.store(speed_kmh, std::memory_order_relaxed);
  state.timestamp_ns.store(timestamp_ns, std::memory_order_relaxed);
  state.sequence.fetch_add(1, std::memory_order_release);
}

LongitudinalVehicleSpeedSnapshot ReadLongitudinalVehicleSpeed() noexcept {
  AtomicVehicleSpeedState& state = GetState();

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

bool IsLongitudinalVehicleSpeedFresh(
    const LongitudinalVehicleSpeedSnapshot& snapshot,
    const std::uint64_t now_ns,
    const std::uint64_t timeout_ns) noexcept {
  if (!snapshot.valid || now_ns < snapshot.timestamp_ns) {
    return false;
  }

  return now_ns - snapshot.timestamp_ns <= timeout_ns;
}

}  // namespace controller
