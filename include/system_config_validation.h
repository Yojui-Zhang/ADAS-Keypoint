#pragma once

#include <string>

struct AdasSystemConfig;

bool ValidateSystemConfig(const AdasSystemConfig& config,
                          std::string* out_error);
