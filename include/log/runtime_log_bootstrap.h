#pragma once

#include <iosfwd>

#include "runtime_log_manager.h"

namespace adas_log {

bool StartAndReportRuntimeLogs(RuntimeLogManager& runtime_log_manager,
                               bool enable_research_logger,
                               bool is_virtual_road,
                               std::ostream& out,
                               std::ostream& err);

}  // namespace adas_log
