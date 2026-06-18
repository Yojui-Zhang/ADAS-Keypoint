# ADAS-Keypoint 使用手冊（重構版）

本版本在**保留 TFLite / TensorRT / SORT / LKA / ACC / Stability / Collision / VehicleBehavior** 既有功能前提下，完成以下整理：

1. 主流程改為「配置驅動」：所有可調參數集中於 `config/system_config.yaml`。
2. 子系統維持獨立責任（SRP）：感知、幾何、控制、碰撞、行為分析分層。
3. 保留舊 API 相容性，新增設定注入口，不破壞原推論與控制流程。

**熱鍵**
熱鍵目前是 1 總 TX、2 油門煞車、3 方向盤、4 推論繪圖、5 ACC、6 Behavior、7 Collision、8 HUD、0 全部繪圖、Backspace 強制關閉所有車控輸出

## 1. 專案模組結構

- `src/main.cpp`
  單一流程協調器：載入配置、初始化子系統、逐幀執行 ADAS pipeline。

- `include/system_config.h`, `src/system_config.cpp`
  集中設定模型與 YAML 載入器。`system_config.yaml` 各欄位映射到對應子系統 Config。

- `src/Camera/input-view.cpp`, `include/Camera/input-view.h`
  輸入來源初始化（檔案/攝影機）、顯示視窗設定（全螢幕、視窗名稱）。

- `Engine/TFlite/*`, `Engine/TensorRT/*`
  推論引擎封裝，保留既有流程。新增 `*_set_sort_config()` 可注入 SORT 與 keypoint filter 設定。

- `src/SORT/*`, `include/SORT/*`
  追蹤與 keypoint 濾波。`SORTTRACKING` 改為可注入 `SortTrackingConfig`。

- `src/Geometry/*`, `include/Geometry/*`
  像素到世界座標轉換。新增 `GeometryConfig`（單位縮放與可視化開關）。

- `src/LKA/*`, `include/LKA/*`
  車道中心線與橫向控制。可用 `lka.lateral_controller` 在 Stanley / MPC 間切換初始 LKA raw steer，並新增 `lane_keeping_set_control_config()`。

- `src/ACC/*`, `include/ACC/*`
  目標選擇與縱向控制（IDM + 濾波 + 限幅），使用 `acc::AccConfig`。

- `src/StabilityControl/*`, `include/StabilityControl/*`
  控制監管層（摩擦圓、舒適性、投影限制），使用 `stability::StabilityConfig`。

- `src/Collision/*`, `include/Collision/*`
  碰撞風險評估與警示輸出。新增 tracker 與 heading fusion 參數可配置。

- `src/VehicleBehavior/*`, `include/VehicleBehavior/*`
  車輛骨架朝向估測與標註。

## 2. 建置與執行

### 2.1 CMake（保持原流程）

```bash
cmake .. -DENGINE=TFLITE -DCANBUS=ITRI
# 或
cmake .. -DENGINE=TENSORRT -DCANBUS=ITRI
make -j$(nproc)
```

### 2.2 執行

```bash
./ADAS <LanePose_Model_Path> <Classify_Model_Path> [System_Config_Path]
```

- 第三參數省略時，程式依序嘗試：
  - `../config/system_config.yaml`
  - `./config/system_config.yaml`
  - `config/system_config.yaml`

範例：

```bash
./ADAS ../weight/pose/model.tflite ../weight/classify/classify.onnx ../config/system_config.yaml
```

## 3. 集中設定檔

- 檔案位置：`config/system_config.yaml`
- 載入器：`LoadSystemConfig()`
- 原則：子系統所有「可標定/可調整」參數集中在這一檔。

### 3.1 頂層區塊

- `app`: 全域執行行為（camera yaml、碰撞是否介入控制、顯示迴圈）。
- `input`: 影像來源與視窗設定。
- `geometry`: 幾何轉換設定。
- `model`: 分類輸入尺寸與 TensorRT 後處理閾值。
- `sort`, `sort_keypoint`: 多目標追蹤與關鍵點濾波。
- `lka`, `acc`, `stability`, `collision`: 控制與風險核心參數。
- `behavior`: 骨架行為分析啟用與索引映射。

## 4. 子系統參數說明（重點）

### 4.1 `app`

- `camera_yaml_path`: 相機內外參檔。
- `fallback_ego_speed_kmh`: 無 CAN 時預設車速。
- `enable_collision_actuation`: 碰撞模組是否直接改寫 speed/steer/brake。
- `wait_key_ms`: `cv::waitKey()` 延遲。

### 4.2 `input`

- `video_path`: OpenCV 影片來源。
- `camera_index`: 攝影機索引（>=0 時優先於 `video_path`）。
- `capture_width`, `capture_height`: 擷取尺寸。
- `window_name`, `fullscreen`: 視窗行為。

### 4.3 `model.tensorrt`

- `topk`: 候選框上限。
- `score_thres`: 置信度門檻。
- `iou_thres`: NMS IoU 門檻。
- `num_labels`: 類別數。

### 4.4 `sort`

- `max_age`: 追蹤器最多失配幀數。
- `min_hits`: 輸出軌跡所需最小連續命中。
- `iou_threshold`: 配對門檻。
- `history_length`: 軌跡歷史長度。

### 4.5 `sort_keypoint`

- `filter_type`: `auto | ema | kf`。
- `allow_env_override`: 是否允許 `SORT_KPT_FILTER` 環境變數覆蓋。
- `ema.*`, `kf.*`: 各濾波器參數。

### 4.6 `geometry`

- `world_unit_scale`: 世界座標單位比例（預設 0.01: cm→m）。
- `draw_kpt_world`, `draw_box_world`: 幾何除錯開關。

### 4.7 `lka`

- 橫向控制器：`lateral_controller` 可設為 `stanley` 或 `mpc`。此開關只改變 LKA 產生的初始橫向轉向值；後續 `StabilitySupervisor` 的摩擦力、離心力、舒適度與速率防護仍共用同一套程式碼。
- Stanley 參數：`k_straight`, `k_curve`, `softening`。
- MPC 參數：`mpc_horizon`, `mpc_q_cte`, `mpc_q_heading`, `mpc_q_steer`, `mpc_r_steer_rate`。
- 參考距離：`x_ref_*`, `x_heading_*`。
- Feedforward：`enable_feedforward`, `ff_gain`, `max_ff_deg`。
- 限幅：`max_steer_deg`, `max_steer_rate_deg_s`, `dt_s`。
- 車道點過濾：`conf_threshold`, `min_x_m`, `max_x_m`, `max_abs_y_m`。
- 曲率模式融合：`metric_*`, `use_hysteresis`, `prob_alpha`。

### 4.8 `acc`

- 目標選擇：`lateral_limit_m`, `min_forward_m`, `max_forward_m`, `lead_hysteresis_m`。
- `lateral_limit_m` 會優先以 LKA 建出的當前車道中心線為基準做 gate；只有落在本車道前方走廊內的車才會進入 ACC lead 候選。若當幀抓不到車道，才退回到以自車座標 `y=0` 的舊邏輯。
- 跟車策略：`cruise_speed_kmh`, `time_gap_s`, `standstill_gap_m`。
- 動態限制：`max_accel_mps2`, `comfort_decel_mps2`, `max_decel_mps2`, `jerk_limit_mps3`。
- 制動映射：`brake_full_decel_mps2`, `brake_multiplier`。

### 4.9 `stability`

- 車輛參數：`mass_kg`, `wheelbase_m`, `steering_ratio`。
- 摩擦與安全：`mu_static`, `mu_dynamic`, `lat_safety`, `total_safety`。
- 舒適與守護：`lat_accel_comfort_mps2`, `long_*_comfort_mps2`, `ttc_hard_guard_s`。
- 速率限制：`max_speed_drop_mps2`, `max_speed_rise_mps2`, `max_jerk_*`。

### 4.10 `collision`

- ROI/走廊：`roi_*`, `corridor_half_width_m`, `danger_forward_m`。
- 風險閾值：`ttc_warn_s`, `ttc_brake_s`, `dis_warn_m`, `dis_brake_m`。
- 警示平滑：`enable_warning_kf`, `warning_kf_*`。
- 追蹤器：`tracker_alpha`, `tracker_beta`, `tracker_*`。
- 骨架融合：`heading_fusion_alpha`。
- 控制介入增益：`max_extra_brake_0_10`, `max_avoid_steer_deg`。

### 4.11 `behavior`

- `enable`: 啟用車輛骨架 heading 模組。
- `use_custom_layout`: 是否使用 `custom_layout`。
- `custom_layout`: 12 點索引映射（top/mid/bot 層級）。

## 5. 主要程式流程

1. `main` 讀取 CLI 與 `system_config.yaml`。
2. 套用配置到 LKA/ACC/Stability/Geometry。
3. 依 `app.run_mode` 切換執行模式：
   - `video`：影片/相機推論主流程
   - `real_car`：實車模式（需 `CANBUS__`）
   - `virtual_road`：純虛擬道路閉迴路模擬（不依賴影片）
4. `video` / `real_car` 每幀執行：
   - 推論 -> SORT
   - Geometry 世界轉換
   - VehicleControl（ACC + LKA + Stability）
   - VehicleBehavior heading
   - Collision warning / actuation
   - UI / video output

## 6. 相容性與注意事項

- 既有 compile-time 巨集（如 `_openCVcap`, `_opengl`, `Write_Video__`）仍保留。
- 新增設定僅擴充注入能力，不移除原演算法邏輯。
- 若配置檔缺少欄位，載入器會使用程式內預設值。

## 7. 建議調參順序

1. `geometry` + `camera_yaml_path`（先確保世界座標可信）
2. `sort` / `sort_keypoint`（追蹤穩定）
3. `lka`（橫向控制）
4. `acc`（縱向控制）
5. `stability`（物理與舒適邊界）
6. `collision`（警示/介入策略）

## 8. 演算法模擬對照（無 CANBus、影片回放）

主流程已新增 `AlgorithmAblationLogger`，呼叫點位於：
- `RunVehicleSkeletonAndHeading(...)` 後
- `Draw info` 前（同幀比較，不改既有控制輸出）

每幀會輸出兩類對照：
1. Skeleton 前後差異：`target_heading_valid/target_heading_deg` 變化。
2. VehicleControl 差異：`VehicleControl_Run` 輸出 vs `ACC+LKA` 原始輸出（不重跑演算法狀態機）。

輸出檔案：
- `ablation_drive_*.csv`：逐幀差異與兩條模擬路徑（VC on / VC off）。
- `ablation_drive_*.summary.txt`：平均與最大差異、路徑偏離統整。
- `ablation_drive_*.csv.route.png`：三線路徑圖（Reference 虛擬道路 / VC on / VC off）。

YAML 開關（建議優先用 `config/system_config.yaml`）：
- `app.run_mode=video|virtual_road|real_car`：切換「跑影片 / 跑虛擬道路模擬 / 跑實車」。
- `ablation.virtual_road_*`：reference 道路來源與參數（含 `mode=csv` + `virtual_road_csv_path`）。
- `ablation.virtual_sim_*`：`run_mode=virtual_road` 時的模擬控制參數（幀數、dt、速度、VC on/off 控制增益）。

環境變數（選用，會覆寫 YAML）：
- `ADAS_ABLATION_LOG_ENABLE=0/1`：關閉或啟用 ablation logger。
- `ADAS_ABLATION_LOG_DIR=/path/to/dir`：設定輸出資料夾。
- `ADAS_ABLATION_LOG_PATH=/path/to/file.csv`：指定完整輸出 CSV 路徑。

虛擬道路驗證（Reference 路徑）：
- `ADAS_ABLATION_VROAD_ENABLE=0/1`：啟用 reference 比對（預設 0）。
- `ADAS_ABLATION_VROAD_MODE=straight|arc|s_curve|csv`：道路型態。
- `ADAS_ABLATION_VROAD_FILE=/path/to/road.csv`：`mode=csv` 時的路徑檔（每列 `x,y`）。
- `ADAS_ABLATION_VROAD_LENGTH_M=300`：內建道路長度（公尺）。
- `ADAS_ABLATION_VROAD_STEP_M=0.5`：內建道路取樣間距（公尺）。
- `ADAS_ABLATION_VROAD_LANE_WIDTH_M=3.5`：車道寬（用於 lane departure 判定）。
- `ADAS_ABLATION_VROAD_ARC_RADIUS_M=120`：`mode=arc` 半徑（公尺，正左負右）。
- `ADAS_ABLATION_VROAD_S_AMPLITUDE_M=2.0`：`mode=s_curve` 振幅（公尺）。
- `ADAS_ABLATION_VROAD_S_WAVELENGTH_M=80`：`mode=s_curve` 波長（公尺）。
