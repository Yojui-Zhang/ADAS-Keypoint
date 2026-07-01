#include "sound/sound.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace sound {
namespace {

constexpr int kAudioDeviceIndex = 3;
constexpr const char* kWarningSoundPath = "./sound/warringsound.wav";
constexpr std::chrono::seconds kWarningCooldown(3);

std::atomic<bool> g_warning_sound_active{false};

}  // namespace

bool RequestLaneDepartureWarningSound() {
  bool expected = false;
  if (!g_warning_sound_active.compare_exchange_strong(expected, true)) {
    return false;
  }

  std::thread([]() {
    char sound_text[256];
    std::snprintf(sound_text,
                  sizeof(sound_text),
                  "aplay -D plughw:%d,0 %s -v",
                  kAudioDeviceIndex,
                  kWarningSoundPath);
    const int ret = std::system(sound_text);
    (void)ret;
    std::this_thread::sleep_for(kWarningCooldown);
    g_warning_sound_active.store(false);
  }).detach();

  return true;
}

}  // namespace sound
