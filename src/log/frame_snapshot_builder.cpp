#include "frame_snapshot_builder.h"

namespace adas_log {

FrameSnapshot BuildFrameSnapshot(const FrameSnapshotBuilderInput& input) {
  FrameSnapshot snapshot;
  snapshot.frame_index = input.frame_index;
  snapshot.frame_sync_ns = input.frame_sync_ns;
  snapshot.frame_hw_ns = input.frame_hw_ns;
  snapshot.cmd_sync_ns = input.cmd_sync_ns;
  snapshot.dt_s = input.dt_s;
  snapshot.ego_speed_kmh = input.ego_speed_kmh;
  snapshot.target_speed_kmh = input.target_speed_kmh;
  snapshot.target_distance_m = input.target_distance_m;
  snapshot.target_ttc_s = input.target_ttc_s;

  if (input.lka_reference_snapshot != nullptr) {
    const LkaReferenceSnapshot& lka = *input.lka_reference_snapshot;
    snapshot.lka_reference_valid = lka.valid;
    snapshot.lka_p_curve = lka.p_curve;
    snapshot.lka_ey_m = lka.ey_m;
    snapshot.lka_epsi_rad = lka.epsi_rad;
    snapshot.lka_mean_kappa_m_inv = lka.mean_kappa_m_inv;
    snapshot.lka_std_kappa_m_inv = lka.std_kappa_m_inv;
    snapshot.lka_current_x_m = lka.current_point.x_m;
    snapshot.lka_current_y_m = lka.current_point.y_m;
    snapshot.lka_target_x_m = lka.target_point.x_m;
    snapshot.lka_target_y_m = lka.target_point.y_m;
  }

  snapshot.lka_current_image_valid = input.lka_current_image_valid;
  snapshot.lka_current_u_px = input.lka_current_px.x;
  snapshot.lka_current_v_px = input.lka_current_px.y;
  snapshot.lka_target_image_valid = input.lka_target_image_valid;
  snapshot.lka_target_u_px = input.lka_target_px.x;
  snapshot.lka_target_v_px = input.lka_target_px.y;

  snapshot.tracking_result = input.tracking_result;
  snapshot.world_before_behavior = input.world_before_behavior;
  snapshot.world_result = input.world_result;
  snapshot.vehicle_cmd = input.vehicle_cmd;
  snapshot.collision_output = input.collision_output;
  snapshot.can_valid = input.can_valid;
  snapshot.can_state = input.can_state;

  if (input.perf != nullptr) {
    const adas_app::RuntimePerformanceMetrics& perf = *input.perf;
    snapshot.perf_fps = perf.fps;
    snapshot.perf_total_ms = perf.total_ms;
    snapshot.perf_input_ms = perf.input_ms;
    snapshot.perf_inference_ms = perf.inference_ms;
    snapshot.perf_geometry_ms = perf.geometry_ms;
    snapshot.perf_acc_scope_ms = perf.acc_scope_ms;
    snapshot.perf_acc_ms = perf.acc_ms;
    snapshot.perf_lka_ms = perf.lka_ms;
    snapshot.perf_stability_ms = perf.stability_ms;
    snapshot.perf_control_total_ms = perf.control_total_ms;
    snapshot.perf_behavior_ms = perf.behavior_ms;
    snapshot.perf_collision_ms = perf.collision_ms;
    snapshot.perf_overlay_ms = perf.overlay_ms;
  }

  return snapshot;
}

}  // namespace adas_log
