#include "keypad.h"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include <fcntl.h>
#include <linux/input.h>
#include <unistd.h>

namespace keypad {

namespace {

user_command_mode_t NormalizeCommand(user_command_mode_t cmd) {
  if (cmd == CMD_DEFAULT) {
    return CMD_NONE;
  }
  return cmd;
}

user_command_mode_t MapAsciiDigit(int key) {
  switch (key) {
    case '0': return CMD_0;
    case '1': return CMD_1;
    case '2': return CMD_2;
    case '3': return CMD_3;
    case '4': return CMD_4;
    case '5': return CMD_5;
    case '6': return CMD_6;
    case '7': return CMD_7;
    case '8': return CMD_8;
    case '9': return CMD_9;
    default: return CMD_DEFAULT;
  }
}

user_command_mode_t MapAsciiLetter(int key) {
  switch (key) {
    case 'q': case 'Q': return CMD_Q;
    case 'w': case 'W': return CMD_W;
    case 'e': case 'E': return CMD_E;
    case 'r': case 'R': return CMD_R;
    case 't': case 'T': return CMD_T;
    case 'y': case 'Y': return CMD_Y;
    case 'u': case 'U': return CMD_U;
    case 'i': case 'I': return CMD_I;
    case 'o': case 'O': return CMD_O;
    case 'p': case 'P': return CMD_P;
    case 'a': case 'A': return CMD_A;
    case 's': case 'S': return CMD_S;
    case 'd': case 'D': return CMD_D;
    case 'f': case 'F': return CMD_F;
    case 'g': case 'G': return CMD_G;
    case 'h': case 'H': return CMD_H;
    case 'j': case 'J': return CMD_J;
    case 'k': case 'K': return CMD_K;
    case 'l': case 'L': return CMD_L;
    case 'z': case 'Z': return CMD_Z;
    case 'x': case 'X': return CMD_X;
    case 'c': case 'C': return CMD_C;
    case 'v': case 'V': return CMD_V;
    case 'b': case 'B': return CMD_B;
    case 'n': case 'N': return CMD_N;
    case 'm': case 'M': return CMD_M;
    default: return CMD_DEFAULT;
  }
}

}  // namespace

struct CommandSource::Impl {
  std::atomic<bool> running{false};
  std::atomic<bool> evdev_ready{false};
  std::atomic<int> pending_command{static_cast<int>(CMD_NONE)};
  std::thread thread;
  ReaderConfig config;
  int fd = -1;
};

CommandSource::CommandSource() : impl_(new Impl()) {}

CommandSource::~CommandSource() {
  Stop();
  delete impl_;
}

bool CommandSource::Start(const ReaderConfig& config) {
  Stop();

  impl_->config = config;
  impl_->pending_command.store(static_cast<int>(CMD_NONE), std::memory_order_release);

  if (config.enable_evdev == false) {
    impl_->running.store(false, std::memory_order_release);
    impl_->evdev_ready.store(false, std::memory_order_release);
    return false;
  }

  const int fd = open(config.device_path.c_str(), O_RDONLY | O_NONBLOCK);
  if (fd < 0) {
    impl_->running.store(false, std::memory_order_release);
    impl_->evdev_ready.store(false, std::memory_order_release);
    impl_->fd = -1;
    return false;
  }

  impl_->fd = fd;
  impl_->running.store(true, std::memory_order_release);
  impl_->evdev_ready.store(true, std::memory_order_release);

  impl_->thread = std::thread([this]() {
    while (impl_->running.load(std::memory_order_acquire)) {
      struct input_event ev {};
      const ssize_t n = read(impl_->fd, &ev, sizeof(ev));
      if (n < static_cast<ssize_t>(sizeof(ev))) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }

      if (ev.type == EV_KEY && ev.value == 1) {
        const user_command_mode_t cmd = NormalizeCommand(MapLinuxKeyCodeToCommand(ev.code));
        if (cmd == CMD_NONE) {
          continue;
        }
        impl_->pending_command.store(static_cast<int>(cmd), std::memory_order_release);
      }
    }
  });
  return true;
}

void CommandSource::Stop() {
  if (impl_ == nullptr) {
    return;
  }

  impl_->running.store(false, std::memory_order_release);

  if (impl_->thread.joinable()) {
    impl_->thread.join();
  }

  if (impl_->fd >= 0) {
    close(impl_->fd);
    impl_->fd = -1;
  }

  impl_->evdev_ready.store(false, std::memory_order_release);
}

void CommandSource::PushCvKey(int key) {
  const user_command_mode_t cmd = NormalizeCommand(MapCvKeyToCommand(key));
  if (cmd == CMD_NONE) {
    return;
  }
  impl_->pending_command.store(static_cast<int>(cmd), std::memory_order_release);
}

user_command_mode_t CommandSource::Consume() {
  const int cmd = impl_->pending_command.exchange(static_cast<int>(CMD_NONE),
                                                  std::memory_order_acq_rel);
  return static_cast<user_command_mode_t>(cmd);
}

bool CommandSource::IsRunning() const {
  return impl_->running.load(std::memory_order_acquire);
}

bool CommandSource::EvdevReady() const {
  return impl_->evdev_ready.load(std::memory_order_acquire);
}

const std::string& CommandSource::DevicePath() const {
  return impl_->config.device_path;
}

user_command_mode_t MapCvKeyToCommand(int key) {
  if (key < 0) {
    return CMD_NONE;
  }

  const user_command_mode_t digit = MapAsciiDigit(key);
  if (digit == CMD_DEFAULT) {
  } else {
    return digit;
  }

  const user_command_mode_t letter = MapAsciiLetter(key);
  if (letter == CMD_DEFAULT) {
  } else {
    return letter;
  }

  switch (key) {
    case 8: return CMD_RETURN;
    case 13: return CMD_ENTER;
    case 45: return CMD_MINUS;
    case 43: return CMD_PLUS;
    case 47: return CMD_SLASH;
    case 44: return CMD_COMMA;
    case 59: return CMD_SEMICOLON;
    default: return CMD_DEFAULT;
  }
}

user_command_mode_t MapLinuxKeyCodeToCommand(int key_code) {
  switch (key_code) {
    case KEY_0:
    case KEY_KP0: return CMD_0;
    case KEY_1:
    case KEY_KP1: return CMD_1;
    case KEY_2:
    case KEY_KP2: return CMD_2;
    case KEY_3:
    case KEY_KP3: return CMD_3;
    case KEY_4:
    case KEY_KP4: return CMD_4;
    case KEY_5:
    case KEY_KP5: return CMD_5;
    case KEY_6:
    case KEY_KP6: return CMD_6;
    case KEY_7:
    case KEY_KP7: return CMD_7;
    case KEY_8:
    case KEY_KP8: return CMD_8;
    case KEY_9:
    case KEY_KP9: return CMD_9;
    case KEY_Q: return CMD_Q;
    case KEY_W: return CMD_W;
    case KEY_E: return CMD_E;
    case KEY_R: return CMD_R;
    case KEY_T: return CMD_T;
    case KEY_Y: return CMD_Y;
    case KEY_U: return CMD_U;
    case KEY_I: return CMD_I;
    case KEY_O: return CMD_O;
    case KEY_P: return CMD_P;
    case KEY_A: return CMD_A;
    case KEY_S: return CMD_S;
    case KEY_D: return CMD_D;
    case KEY_F: return CMD_F;
    case KEY_G: return CMD_G;
    case KEY_H: return CMD_H;
    case KEY_J: return CMD_J;
    case KEY_K: return CMD_K;
    case KEY_L: return CMD_L;
    case KEY_Z: return CMD_Z;
    case KEY_X: return CMD_X;
    case KEY_C: return CMD_C;
    case KEY_V: return CMD_V;
    case KEY_B: return CMD_B;
    case KEY_N: return CMD_N;
    case KEY_M: return CMD_M;
    case KEY_BACKSPACE: return CMD_RETURN;
    case KEY_ENTER:
    case KEY_KPENTER: return CMD_ENTER;
    case KEY_MINUS:
    case KEY_KPMINUS: return CMD_MINUS;
    case KEY_EQUAL:
    case KEY_KPPLUS: return CMD_PLUS;
    case KEY_SLASH:
    case KEY_KPSLASH: return CMD_SLASH;
    case KEY_COMMA: return CMD_COMMA;
    case KEY_SEMICOLON: return CMD_SEMICOLON;
    default: return CMD_DEFAULT;
  }
}

}  // namespace keypad
