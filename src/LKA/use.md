
只要確保 TrackingBox.kpts 在呼叫前已經是 (x_m, y_m, conf) 的車體地面座標即可：

ControlConfig cfg;
cfg.wheel_base_m = 0.30f;
cfg.velocity_mps = 5.0f;
cfg.dt_s = 0.02f;

cfg.k_straight = 0.7f;
cfg.k_curve = 3.0f;

// 直線看遠、彎道看近
cfg.x_heading_straight_m = 1.5f;
cfg.x_heading_curve_m = 0.8f;

// 機率平滑
cfg.enable_prob_lowpass = true;
cfg.prob_alpha = 0.85f;

// 若你覺得 sigmoid 仍會抖，改用 hysteresis
// cfg.use_sigmoid_probability = false;
// cfg.use_hysteresis = true;

ControlState st;

TrackingBox lane;
lane.class_id = 0;
// lane.kpts = 你已經過濾+homography後的車體座標點

float steer_deg = calculate_lane_steering(lane, cfg, &st);
// st.debug 可直接拿去 log

