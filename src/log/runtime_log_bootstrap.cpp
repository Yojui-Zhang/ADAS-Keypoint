#include "runtime_log_bootstrap.h"

#include <ostream>
#include <string>

namespace adas_log {

bool StartAndReportRuntimeLogs(RuntimeLogManager& runtime_log_manager,
                               bool enable_research_logger,
                               bool is_virtual_road,
                               std::ostream& out,
                               std::ostream& err) {
  std::string log_error;
  if (runtime_log_manager.Start(enable_research_logger, &log_error) == false) {
    err << "Main: " << log_error << std::endl;
    return false;
  }

  if (runtime_log_manager.AblationRunning()) {
    out << "Main: Ablation log -> " << runtime_log_manager.AblationOutputPath() << std::endl;
  } else {
    out << "Main: Ablation logger disabled." << std::endl;
  }

  if (is_virtual_road == false) {
    if (runtime_log_manager.ResearchRunning()) {
      out << "Main: Research log -> " << runtime_log_manager.ResearchOutputPath() << std::endl;
    } else {
      out << "Main: Research logger disabled." << std::endl;
    }
  }

  return true;
}

}  // namespace adas_log
