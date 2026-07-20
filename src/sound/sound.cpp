#include "sound/sound.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace sound {
namespace {

constexpr int kAudioDeviceIndex = 0;
constexpr const char* kLaneDepartureWarningSoundPath = "../sound/dondondon-01.wav";
constexpr const char* kAebWarningSoundPath = "../sound/beeeeeeee-01.wav";
constexpr std::chrono::seconds kWarningCooldown(3);

std::atomic<bool> g_lane_departure_warning_sound_active{false};
std::atomic<bool> g_aeb_warning_sound_active{false};

bool RequestWarningSound(const char* sound_path,
                         std::atomic<bool>& active_flag) {
  bool expected = false;
  if (!active_flag.compare_exchange_strong(expected, true)) {
    return false;
  }

  std::thread([sound_path, &active_flag]() {
    char sound_text[256];
    std::snprintf(sound_text,
                  sizeof(sound_text),
                  "aplay -D plughw:%d,0 %s -v",
                  kAudioDeviceIndex,
                  sound_path);
    const int ret = std::system(sound_text);
    (void)ret;
    std::this_thread::sleep_for(kWarningCooldown);
    active_flag.store(false);
  }).detach();

  return true;
}

}  // namespace

bool RequestLaneDepartureWarningSound() {
  return RequestWarningSound(kLaneDepartureWarningSoundPath,
                             g_lane_departure_warning_sound_active);
}

bool RequestAebWarningSound() {
  return RequestWarningSound(kAebWarningSoundPath,
                             g_aeb_warning_sound_active);
}

}  // namespace sound
