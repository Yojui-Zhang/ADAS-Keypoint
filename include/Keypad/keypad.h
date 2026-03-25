#pragma once

#include <string>

#include "user_command.h"

namespace keypad {

struct ReaderConfig {
  bool enable_evdev = true;
  std::string device_path = "/dev/input/event1";
};

class CommandSource {
public:
  CommandSource();
  ~CommandSource();

  bool Start(const ReaderConfig& config = ReaderConfig{});
  void Stop();

  void PushCvKey(int key);
  user_command_mode_t Consume();

  bool IsRunning() const;
  bool EvdevReady() const;
  const std::string& DevicePath() const;

private:
  struct Impl;
  Impl* impl_;
};

user_command_mode_t MapCvKeyToCommand(int key);
user_command_mode_t MapLinuxKeyCodeToCommand(int key_code);

}  // namespace keypad
