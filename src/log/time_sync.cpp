#include "time_sync.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <fcntl.h>
#include <time.h>
#include <unistd.h>

namespace {

#ifndef CLOCKFD
#define CLOCKFD 3
#endif

#ifndef FD_TO_CLOCKID
#define FD_TO_CLOCKID(fd) ((~(clockid_t)(fd) << 3) | CLOCKFD)
#endif

struct TimeSyncState {
  int ptp_fd = -1;
  clockid_t clock_id = CLOCK_REALTIME;
  bool using_ptp = false;
  bool init_ok = true;
  std::string source = "CLOCK_REALTIME";
  std::string init_error;
  bool initialized = false;
};

TimeSyncState& GetState() {
  static TimeSyncState state;
  return state;
}

std::once_flag& GetInitOnceFlag() {
  static std::once_flag flag;
  return flag;
}

std::atomic<uint64_t>& CanSteerTxNs() {
  static std::atomic<uint64_t> ts_ns{0};
  return ts_ns;
}

std::atomic<uint64_t>& CanBrakeTxNs() {
  static std::atomic<uint64_t> ts_ns{0};
  return ts_ns;
}

bool ParseBoolEnv(const char* value, bool default_value) {
  if (!value) return default_value;

  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  return default_value;
}

uint64_t ClockNowNs(clockid_t clock_id) {
  struct timespec ts {};
  if (clock_gettime(clock_id, &ts) != 0) {
    return 0;
  }
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

void InitializeTimeSync() {
  TimeSyncState& state = GetState();
  if (state.initialized) return;

  const char* ptp_dev_env = std::getenv("ADAS_PTP_DEVICE");
  const std::string ptp_device = (ptp_dev_env && *ptp_dev_env) ? std::string(ptp_dev_env) : std::string("/dev/ptp0");

  const char* ptp_required_env = std::getenv("ADAS_PTP_REQUIRED");
  const bool ptp_required = ParseBoolEnv(ptp_required_env, false);

  int fd = open(ptp_device.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd >= 0) {
    const clockid_t ptp_clock_id = FD_TO_CLOCKID(fd);
    if (ClockNowNs(ptp_clock_id) > 0) {
      state.ptp_fd = fd;
      state.clock_id = ptp_clock_id;
      state.using_ptp = true;
      state.init_ok = true;
      state.source = ptp_device;
      state.initialized = true;
      return;
    }
    close(fd);
    fd = -1;
  }

  state.ptp_fd = -1;
  state.clock_id = CLOCK_REALTIME;
  state.using_ptp = false;
  state.source = "CLOCK_REALTIME";

  if (ptp_required) {
    state.init_ok = false;
    state.init_error = "PTP required but unavailable. Set ADAS_PTP_DEVICE to a valid /dev/ptpX, or unset ADAS_PTP_REQUIRED.";
  } else {
    state.init_ok = true;
    state.init_error.clear();
  }

  state.initialized = true;
}

void EnsureInitialized() {
  std::call_once(GetInitOnceFlag(), InitializeTimeSync);
}

}  // namespace

bool TimeSyncInit(std::string* out_error) {
  EnsureInitialized();
  const TimeSyncState& state = GetState();

  if (!state.init_ok && out_error) {
    *out_error = state.init_error;
  }
  return state.init_ok;
}

uint64_t TimeSyncNowNs() {
  EnsureInitialized();
  const TimeSyncState& state = GetState();

  uint64_t ts_ns = ClockNowNs(state.clock_id);
  if (ts_ns != 0) return ts_ns;

  ts_ns = ClockNowNs(CLOCK_REALTIME);
  return ts_ns;
}

bool TimeSyncUsingPtp() {
  EnsureInitialized();
  return GetState().using_ptp;
}

const std::string& TimeSyncClockSource() {
  EnsureInitialized();
  return GetState().source;
}

void TimeSyncMarkCanSteerTxNs(uint64_t ts_ns) {
  CanSteerTxNs().store(ts_ns, std::memory_order_release);
}

void TimeSyncMarkCanBrakeTxNs(uint64_t ts_ns) {
  CanBrakeTxNs().store(ts_ns, std::memory_order_release);
}

uint64_t TimeSyncGetCanSteerTxNs() {
  return CanSteerTxNs().load(std::memory_order_acquire);
}

uint64_t TimeSyncGetCanBrakeTxNs() {
  return CanBrakeTxNs().load(std::memory_order_acquire);
}
