# AGENT 工作紀錄與架構摘要

## 1. 本文件目的
- 彙整本次對話的需求、重點決策、架構調整與驗證結果。
- 作為後續維護與交接時的快速索引文件。

## 2. 用戶要求（分類）
1. 在不刪除子系統功能前提下進行重構整理（SOLID 與工程可維護性）。
2. 將系統可調參數集中至單一控制檔（原 config 類型）。
3. 在 `README.md` 撰寫整理後模組與可調變數的使用手冊。
4. 追加本文件：整理本次對話重點、架構概要、需求與結論。

## 3. 歷史對話重點（濃縮）
- 初始需求：保留 TFLite、TensorRT、SORT、ACC、LKA、Geometry、Collision 等功能，進行程式工程化整理與參數集中化。
- 執行方式：先盤點現有參數來源與硬編碼值，再建立集中設定層，最後將各模組配置注入點接上。
- 產出結果：
  - 新增集中設定模型與載入器：`include/system_config.h`、`src/system_config.cpp`
  - 新增統一配置檔：`config/system_config.yaml`
  - 主流程改為配置驅動：`src/main.cpp`
  - 使用手冊更新：`README.md`
- 本輪追加需求：將上述內容整理分類並新增 `AGENT.md`。

## 4. 思考與決策摘要（可公開版本）
- **決策 A：以配置注入取代散落常數**
  - 原因：避免參數分散於多模組與巨集，降低維護成本與調參風險。
  - 做法：建立 `AdasSystemConfig`，由 `system_config.yaml` 統一載入後分發至各子系統。

- **決策 B：保留舊流程相容，新增可選注入接口**
  - 原因：降低回歸風險，確保既有推論與控制流程可持續使用。
  - 做法：在 LKA/Geometry/SORT/Collision/Engine 追加 `SetConfig` 或對應配置 API，不移除原核心算法。

- **決策 C：主流程責任拆分**
  - 原因：提升可讀性與可測試性，便於後續擴充。
  - 做法：將 CLI、配置載入、初始化、每幀處理分段，避免單一函式過度耦合。

## 5. 架構概要（重構後）

### 5.1 流程層
- `src/main.cpp`
  - 負責：CLI -> 載入 `system_config.yaml` -> 初始化引擎與子系統 -> 逐幀執行 pipeline。

### 5.2 配置層
- `include/system_config.h`, `src/system_config.cpp`
  - 負責：定義統一配置結構與 YAML 解析。
- `config/system_config.yaml`
  - 負責：集中化調參（app/input/model/sort/lka/acc/stability/collision/behavior）。

### 5.3 子系統層（主要可注入點）
- LKA：`lane_keeping_set_control_config(...)`
- Geometry：`Geometry_SetConfig(...)`
- ACC：`acc::ACC_SetConfig(...)`
- Stability：`stability::VehicleControl_SetStabilityConfig(...)`
- Collision：`collision::CollisionAssistConfig` + `SetConfig(...)`
- SORT：
  - 追蹤器參數：`SORTTRACKING::SortTrackingConfig`
  - Keypoint 濾波：`sort_kpt::KeypointFilterConfig`
- 引擎入口：
  - TFLite：`tflite_set_sort_config(...)`
  - TensorRT：`trt_set_sort_config(...)`

## 6. 主要改動摘要（檔案分組）

### 6.1 新增檔案
- `config/system_config.yaml`
- `include/system_config.h`
- `src/system_config.cpp`
- `README.md`（重寫為使用手冊）
- `AGENT.md`（本文件）

### 6.2 既有檔案調整
- 主流程：`src/main.cpp`
- 輸入配置化：`include/Camera/input-view.h`, `src/Camera/input-view.cpp`
- Geometry 配置化：`include/Geometry/GeometryFunction.h`, `src/Geometry/GeometryFunction.cpp`
- LKA 配置注入：`include/LKA/lane_keeping.h`, `src/LKA/lane_keeping.cpp`
- SORT 配置化：
  - `include/SORT/SortTracking.h`, `src/SORT/SortTracking.cc`
  - `include/SORT/KeypointFilterSwitch.h`, `src/SORT/KeypointFilterSwitch.cc`
- Collision 擴充調參：
  - `include/Collision/CollisionAssistApi.h`, `src/Collision/CollisionAssistApi.cpp`
- 引擎注入點：
  - `Engine/TFlite/include/TFlite_main.h`, `Engine/TFlite/src/TFlite_main.cc`
  - `Engine/TensorRT/include/TensorRT_main.hpp`, `Engine/TensorRT/src/TensorRT_main.cc`
- 編譯自包含修正：`include/write_video.h`

## 7. 驗證與結論
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
  - `cmake --build build-TensorRT -j4`：通過（存在第三方 TensorRT deprecated warnings，非本次功能性錯誤）
- 執行參數驗證：
  - `./build-TFlite/ADAS`（無參數）可正確顯示新用法
- 結論：
  - 本次需求已完成：功能保留、參數集中、手冊補齊、架構明確化。

## 8. 已知限制與風險
- 專案仍保留部分 compile-time 巨集（如 `_openCVcap`、`Write_Video__`）；目前採「增量式配置化」，未全面去巨集化。
- 第三方 TensorRT API 有 deprecation warning，屬外部依賴版本議題，非本輪重構主題。

## 9. 後續建議
1. 增加 `system_config.yaml` schema/欄位驗證與錯誤提示（避免拼字錯誤無感）。
2. 針對關鍵子系統（LKA/ACC/Collision）補最小單元測試與回歸測試樣本。
3. 視需求逐步把 compile-time 巨集轉為 runtime config，進一步統一行為控制。

---

## 10. 本輪追加（論文級數據紀錄與時間同步）

### 10.1 本輪用戶要求（分類）
1. 在不改動既有演算法函式核心邏輯前提下，新增論文級資料紀錄機制。
2. 需紀錄至少 `angle`、`speed` 等關鍵控制與車態資料。
3. 可選擇加入可推導行徑路線的資訊（例如 CANBus / controller 可得訊號）。
4. 影像幀時間戳與 CANBus 轉向/煞車命令時間戳需嚴格對齊；優先使用 Hardware Trigger / PTP。

### 10.2 歷史對話重點（本輪濃縮）
- 先盤點資料來源：`src/main.cpp` 已有控制輸出（LKA/ACC/Collision），`CANBus/itri` 已有速度、轉角、yaw、theta、accel 等。
- 第一階段先建立獨立 logger 模組，不侵入現有控制演算法。
- 用戶追加「嚴格對齊」要求後，升級為同一時鐘域架構：引入 PTP 優先時間基準，並在影像採集點、命令定案點、CAN 實際發送點各自打時間戳。
- 最終保留原流程，新增研究資料層與時間同步層。

### 10.3 思考與決策摘要（可公開版本）
- **決策 D：採外掛式記錄層，不修改演算法核心**
  - 原因：用戶要求不可動原本函式與邏輯，且需降低回歸風險。
  - 做法：新增 `ResearchDataLogger` + `TimeSync`，僅在主流程與 I/O 發送點掛載資料採樣。

- **決策 E：同一時鐘域記錄（PTP 優先）**
  - 原因：論文級時間對齊需可證明同源時鐘，避免跨時鐘域直接比較。
  - 做法：`frame_sync_ns`、`cmd_sync_ns`、`can_*_tx_sync_ns` 一律使用 `TimeSyncNowNs()`；PTP 不可用時明確標示 fallback。

- **決策 F：硬體幀時間戳獨立保留**
  - 原因：V4L2 driver 的 `buf.timestamp` 可能與控制時鐘域不同，不應混為同欄位。
  - 做法：另存 `frame_hw_ns`，供離線校正/比對，不直接取代同步欄位。

### 10.4 架構概要（本輪新增）

#### 10.4.1 時間同步層
- `include/time_sync.h`, `src/time_sync.cpp`
  - 功能：初始化同步時鐘、提供統一 `ns` 時間戳、保存最近 CAN steer/brake TX 時間戳。
  - PTP 行為：
    - 預設嘗試 `/dev/ptp0`
    - `ADAS_PTP_DEVICE` 可指定裝置
    - `ADAS_PTP_REQUIRED=1` 時，若無 PTP 則啟動失敗

#### 10.4.2 研究資料層
- `include/research_data_logger.h`, `src/research_data_logger.cpp`
  - 功能：每幀輸出 CSV 與 summary。
  - 核心時間欄位：
    - `frame_sync_ns`（同時鐘域）
    - `frame_hw_ns`（V4L2 DQBUF 原始硬體時間）
    - `cmd_sync_ns`（命令定案）
    - `can_steer_tx_sync_ns`、`can_brake_tx_sync_ns`（實際 write 後）
  - 補充欄位：
    - 控制：`cmd_speed_kmh/cmd_steer_deg/cmd_brake_0_10`
    - ACC/Collision 指標：`TTC`、距離、威脅目標
    - CAN 車態：`speed/steer/yaw/theta/accel/torque/meterage/...`
    - 路徑估計：`route_x_m/route_y_m/route_heading_rad/route_distance_m`

#### 10.4.3 接線層（不改演算法核心）
- `src/main.cpp`
  - 在影像取得後記錄 `frame_sync_ns`
  - 在控制命令定案後記錄 `cmd_sync_ns`
  - 每幀收集 world/CAN/控制數據寫入 logger
- `CANBus/itri/src/canbus_recv.cpp`
  - 在 steer/brake frame `write()` 成功後更新 TX 時間戳
- `src/Camera/V4L2_define.cpp`
  - 擷取 `VIDIOC_DQBUF` 的 `buf.timestamp` 並提供讀取函式

### 10.5 本輪主要改動摘要（檔案分組）

#### 10.5.1 新增檔案
- `include/time_sync.h`
- `src/time_sync.cpp`
- `include/research_data_logger.h`
- `src/research_data_logger.cpp`

#### 10.5.2 既有檔案調整
- `src/main.cpp`
- `CANBus/itri/src/canbus_recv.cpp`
- `src/Camera/V4L2_define.cpp`

### 10.6 驗證與結論（本輪）
- 建置驗證：
  - `cmake -S . -B build-TFlite`：通過
  - `cmake --build build-TFlite -j4`：通過
- 結論：
  - 本輪需求已完成：新增論文級資料紀錄機制，並提供 PTP 優先的幀/命令/CAN TX 對齊時間戳，且未改動既有演算法核心控制流程。

### 10.7 已知限制與風險（本輪）
- `_openCVcap` 路徑無原生硬體幀時間戳，`frame_sync_ns` 為擷取後同時鐘域取樣時間。
- `can_*_tx_sync_ns` 目前為 userspace `write()` 成功時間，非硬體控制器回報的 bus-level ACK 時間。
- 若未啟用 `ADAS_PTP_REQUIRED=1`，PTP 不可用時會 fallback 至 `CLOCK_REALTIME`，需在論文中標註時鐘來源。
- `frame_hw_ns` 與 `frame_sync_ns` 可能分屬不同時鐘域；做嚴格統計前需先校時或建模偏移。

### 10.8 後續建議（本輪）
1. 在實車部署端加上 PHC/系統時鐘校時監控（`phc2sys`/`ptp4l`）並定期存證 offset。
2. 若硬體支援，改為網卡/控制器硬體 TX timestamp（SO_TIMESTAMPING）以提升命令時間戳可信度。
3. 新增離線分析腳本（時間對齊誤差、控制延遲分佈、事件對齊統計）作為論文附錄材料。

---

## 11. 本輪追加（演算法模擬對照層與路徑圖）

### 11.1 本輪用戶要求（分類）
1. 在不大改既有程式與演算法核心前提下，於 `RunVehicleSkeletonAndHeading(...)` 後、`Draw info` 前加入比較函式。
2. 比較「有/無 `RunVehicleSkeletonAndHeading`」與「有/無 `VehicleControl_Run`（控制層）」差異。
3. 依演算法結果輸出可做論文分析的圖表/統整，並產生模擬實際上路的路徑圖。
4. 該模擬函式以「無 CANBus、跑影片」場景為主；後續實車再以 CANBus + 即時影像與 `log_frame` 比對。

### 11.2 歷史對話重點（本輪濃縮）
- 先定位主流程插入點（`src/main.cpp` 內 `VehicleControl_Run`、`RunVehicleSkeletonAndHeading`、`DrawTargetInfo` 區段）。
- 盤點 `VehicleControl_Run`、ACC、LKA、Stability stateful 行為後，確認不可直接重跑「第二次 VehicleControl_Run」作 ablation，避免污染原本控制狀態。
- 採用最小侵入策略：新增獨立 logger 模組，在同幀只讀取既有輸出做比較與模擬，不改原控制鏈。

### 11.3 思考與決策摘要（可公開版本）
- **決策 G：避免重跑 stateful 控制鏈，改做同幀對照分析**
  - 原因：`VehicleControl_Run` 內含 ACC/LKA/Supervisor 狀態遞推，重跑會影響主流程結果，違反「盡量維持原碼行為」。
  - 做法：以同一幀 `cmd` 中的 `acc_cmd` + `lka_steer_deg_raw` 作為「VC off」基準，與 `VehicleControl_Run` 輸出（VC on）比較。

- **決策 H：Skeleton 比較採前後 world_result 快照**
  - 原因：`RunVehicleSkeletonAndHeading` 會就地更新 `target_heading_valid/target_heading_deg`。
  - 做法：在呼叫前複製 `world_result`，呼叫後比對同 ID 物件 heading valid/角度差異。

- **決策 I：輸出論文友善三件套（CSV + summary + route.png）**
  - 原因：需可追溯、可量化、可視覺化。
  - 做法：每幀記錄差異欄位，結束輸出摘要，並畫出 VC on / VC off 雙路徑圖。

### 11.4 架構概要（本輪新增）

#### 11.4.1 對照資料層
- `include/algorithm_ablation_logger.h`, `src/algorithm_ablation_logger.cpp`
  - 功能：每幀計算 Skeleton 前後差異、VC on/off 差異、路徑積分與輸出。
  - 主要輸出：
    - `ablation_drive_*.csv`
    - `ablation_drive_*.summary.txt`
    - `ablation_drive_*.csv.route.png`

#### 11.4.2 主流程接線層
- `src/main.cpp`
  - 初始化 `AlgorithmAblationLogger`。
  - 在 `RunVehicleSkeletonAndHeading(...)` 後、`DrawTargetInfo(...)` 前呼叫 `ablation_logger.Step(...)`。
  - 結束時呼叫 `ablation_logger.Stop()`。

### 11.5 主要改動摘要（檔案分組）

#### 11.5.1 新增檔案
- `include/algorithm_ablation_logger.h`
- `src/algorithm_ablation_logger.cpp`

#### 11.5.2 既有檔案調整
- `src/main.cpp`
- `README.md`（新增演算法模擬對照章節與環境變數說明）

### 11.6 驗證與結論（本輪）
- 建置驗證：
  - `cmake -S . -B build-TFlite`：通過
  - `cmake --build build-TFlite -j4`：通過
- 執行參數驗證：
  - `./build-TFlite/ADAS`（無參數）可正常顯示 usage
- 結論：
  - 已完成「不改核心邏輯的模擬對照層」，可在影片模式輸出對照路徑圖與對應數據檔。

### 11.7 已知限制與風險（本輪）
- 「VC off」為同幀 `ACC+LKA raw` 對照，不是重跑一條獨立控制狀態機；優點是不污染主流程，代價是無法覆蓋二次遞推效應。
- `CANBUS__` 編譯路徑下 `AlgorithmAblationLogger` 預設關閉（可用環境變數覆寫啟用）。

---

## 12. 本輪再追加（Luxgen 參數修正與實車開關問答）

### 12.1 本輪用戶要求（分類）
1. 修改 `system_config.yaml` 車輛參數，要求參考網路可查的 Luxgen S3 EV/S3 EV+資料。
2. 後續僅回答問題，不再修改程式碼：說明實車測試要開哪些開關，才能「控車 + 即時影像 + 類模擬的圖片/數據輸出」。

### 12.2 思考與決策摘要（可公開版本）
- **決策 J：車身幾何採公開一致值，維持軸距 2.62 m**
  - 原因：S3 車系公開規格對軸距資訊一致，風險低。

- **決策 K：`mass_kg` 採工程估計並明確註記**
  - 原因：S3 EV+缺少穩定可追溯的官方整備重公開值。
  - 做法：以 S3 ICE 約 1230~1250 kg 為基準，設定 `1350 kg` 作 EV 化估計值，並在 YAML 註解標示為估算。

- **決策 L：`steering_ratio` 先維持 1.0**
  - 原因：目前控制鏈路中 `steer_deg` 以路輪角語意運作；直接改成方向盤轉向比（例如 14.5）會導致量級錯置。
  - 做法：先不改，待後續若切換到方向盤角命令語意，再整體重構對齊。

### 12.3 本輪實際修改
- `config/system_config.yaml`
  - `stability.mass_kg: 1500.0 -> 1350.0`
  - 補充 `lka.wheel_base_m` / `stability.wheelbase_m` 來源註解
  - 補充 `stability.steering_ratio` 維持 1.0 的語意註解

### 12.4 外部資料依據（2026-02-25 查詢）
- Luxgen S3 規格（軸距 2620mm、車重區間可參考）：  
  `https://auto.ltn.com.tw/attr/spec/1878/1880`  
  `https://auto.ltn.com.tw/attr/spec/1878/1882`
- S3 EV/S3 EV+相關資料與量產脈絡：  
  `https://news.u-car.com.tw/article/35498`  
  `https://am.u-car.com.tw/article/62013`

### 12.5 已回答之實車測試開關清單（不改碼版）
- 編譯/巨集：
  - `CANBUS__`：開啟控車
  - 影像來源二選一：`_openCVcap` 或 `_v4l2cap`
  - `Write_Video__`：輸出 `Output_video.mp4`
  - CMake `-DCANBUS=ITRI|YUNTECH` 對應實車 CAN 堆疊
- 執行期環境變數：
  - `ADAS_RESEARCH_LOG_ENABLE=1`、`ADAS_RESEARCH_LOG_DIR=...`
  - `ADAS_ABLATION_LOG_ENABLE=1`、`ADAS_ABLATION_LOG_DIR=...`
  - PTP 嚴格模式：`ADAS_PTP_REQUIRED=1`、`ADAS_PTP_DEVICE=/dev/ptpX`
- YAML：
  - `input.camera_index >= 0` 使用即時攝影機
  - `app.enable_collision_actuation` 依實測策略開關

### 12.6 結論（本輪）
- 已完成車輛參數 YAML 的 Luxgen 依據化修正（含估算註記），並提供實車測試所需開關的完整清單。
- 目前架構可同時支援：
  - 即時影像顯示/錄影
  - 研究 CSV + summary
  - 模擬對照路徑圖（ablation）

---

## 13. 本輪再追加（三模式切換與三線論文比較圖）

### 13.1 本輪用戶要求（分類）
1. 圖表需求：論文比較圖 (`.route.png`) 需固定顯示三條線：`虛擬道路`、`VC on`、`VC off`。
2. 行為需求：指出目前「跑影片」時 VC 路徑與虛擬道路無直接耦合，要求可切換不同執行策略。
3. 開關需求：希望可切換三種運行模式：
   - 跑虛擬道路模擬
   - 跑影片
   - 跑實車
4. 文件需求：將本次思考摘要、結論、架構與歷史重點整理回寫至 `AGENT.md`。

### 13.2 思考與決策摘要（可公開版本）
- **決策 M：新增 `run_mode` 做流程分流，而非硬塞在同一主迴圈**
  - 原因：`video` 與 `virtual_road` 的資料來源本質不同；混在同流程會造成語意混亂。
  - 做法：`app.run_mode=video|virtual_road|real_car`，在 `main` 初始化後直接分流。

- **決策 N：新增「純虛擬道路閉迴路模擬」分支**
  - 原因：用戶需要可控、可重現、與 reference road 直接耦合的論文比較。
  - 做法：在 `AlgorithmAblationLogger` 增加 `RunVirtualRoadSimulation(...)`，由 reference road 投影誤差驅動 VC on/off 兩條路徑，保證三線圖可比較。

- **決策 O：保留既有影片/實車流程，最小侵入改造**
  - 原因：避免破壞原本推論、控制、顯示與 CAN 接線行為。
  - 做法：`video/real_car` 保持原 pipeline；只在 `virtual_road` 模式直接執行模擬並輸出 CSV/PNG 後結束。

- **決策 P：道路 CSV 路徑改為可相對配置檔解析**
  - 原因：避免不同啟動目錄導致 `./road_csv/...` 找不到檔案。
  - 做法：新增路徑解析邏輯，優先支援相對於 config 檔路徑與工作目錄。

### 13.3 架構概要（本輪新增）

#### 13.3.1 執行模式層
- `AppRuntimeConfig.run_mode`
  - `video`：原影片/相機推論流程（含 VehicleControl、Behavior、Collision、Draw）。
  - `virtual_road`：不啟動影像推論，直接做 reference road 閉迴路模擬並輸出三線圖。
  - `real_car`：實車模式；若未啟用 `CANBUS__`，會提示並回退 `video`。

#### 13.3.2 Ablation 模擬層
- 新增 `VirtualRoadSimulationOptions` 與 `RunVirtualRoadSimulation(...)`：
  - 可配置幀數、`dt`、模擬速度、最大轉角。
  - 可分別設定 VC on / VC off 控制增益與 VC off 偏置/擾動，形成可辨識對照曲線。
- 輸出維持：
  - `ablation_drive_*.csv`
  - `ablation_drive_*.summary.txt`
  - `ablation_drive_*.csv.route.png`（三條線）

#### 13.3.3 配置層擴充
- `ablation` 區塊新增 `virtual_sim_*` 參數群（模擬控制器）。
- 預設 `virtual_road_mode=csv`，並讀取 `./road_csv/s_curve.csv`。

### 13.4 本輪主要改動檔案
- 執行模式與流程分流：
  - `src/main.cpp`
- 配置結構與載入：
  - `include/system_config.h`
  - `src/system_config.cpp`
  - `config/system_config.yaml`
- Ablation 模擬功能：
  - `include/algorithm_ablation_logger.h`
  - `src/algorithm_ablation_logger.cpp`
- 文件更新：
  - `README.md`

### 13.5 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
- 執行驗證（`run_mode=virtual_road`）：
  - 程式可正常完成模擬並輸出：
    - `research_logs/ablation_drive_*.csv`
    - `research_logs/ablation_drive_*.csv.summary.txt`
    - `research_logs/ablation_drive_*.csv.route.png`
  - summary 可見 `virtual_road_active=1` 且具備 VC on/off 對 reference road 的 CTE/heading 統計。

### 13.6 使用方式（本輪最終）
1. 跑虛擬道路論文圖（建議）
   - `app.run_mode: "virtual_road"`
   - `ablation.virtual_road_enable: 1`
   - `ablation.virtual_road_mode: "csv"`
   - `ablation.virtual_road_csv_path: "./road_csv/s_curve.csv"`
2. 跑影片
   - `app.run_mode: "video"`
3. 跑實車
   - `app.run_mode: "real_car"` + 編譯開啟 `CANBUS__`

### 13.7 結論（本輪）
- 已完成三線比較圖目標（Reference / VC on / VC off）與三模式切換設計。
- 「跑影片」與「跑虛擬道路」已明確拆分，不再語意混用。
- 保留原控制與推論主流程，新增能力以最小侵入方式落地。

---

## 14. 本輪再追加（Keypad 執行期控制、CAN TX 安全拆分、HUD 與 log 模組整理）

### 14.1 本輪用戶要求（分類）
1. 依 `./src/Keypad/` 與 `./include/Keypad/` 新增按鍵控制功能。
2. 將 `CANBUS__` 控制策略改為預設「僅接收 CAN、不發送控車訊號」。
3. 新增可分離的執行期開關，分開控制：
   - CAN TX 總開關
   - 油門 / 煞車控制
   - 方向盤控制
4. 方向盤每次啟用時，必須依 `0x201` 送出 `0x02 -> 0x03 -> 0x03` 初始化流程，確保可重新接管。
5. 在畫面上新增文字顯示目前油門、煞車、方向盤等控制開關狀態。
6. 新增按鍵開關控制各類演算法繪圖，且需分開控制：
   - `trt_process_frame` / `tflite_run_frame`
   - `ACC_DrawTrackingBoxes`
   - `RunVehicleSkeletonAndHeading`
   - Collision overlay
7. 將 `main` 中的 log 初始化與每幀紀錄整理到 `./src/log`、`./include/log`，讓 `main` 更乾淨。
8. 後續僅回答問題，不再改碼時，需能說明：
   - HUD 文字位置在哪裡改
   - `TX` 熱鍵的實際語意
   - `CAN compile` 與 `keypad` HUD 狀態欄位代表什麼

### 14.2 歷史對話重點（本輪濃縮）
- 用戶要求把控車行為從「編譯期/啟動期固定」改為「執行期熱鍵切換」，並把 CAN 接收與 CAN 控車發送拆開。
- 為避免誤控車，改成預設 receive-only：程式啟動後只做 `canbus_recv(...)`，不直接啟動方向盤或油門煞車發送執行緒。
- 新增 keypad runtime control 狀態，熱鍵可即時切換總 TX、縱向控制、方向盤控制，以及各種演算法繪圖。
- 方向盤重新啟用時，不直接開 thread，而是先補送 EPS 初始化序列，再進入 steering control thread。
- 為回應「main 要乾淨」的要求，將 logger 啟動、virtual road 執行入口、每幀 research/ablation 封包搬到 `src/log` / `include/log` 的 wrapper。
- 對話後段補充澄清：
  - `TX master` 不是控制是否接收 CAN，而是是否允許發送控車 CAN。
  - `CAN compile` 不是在顯示 CAN 線上有沒有資料，而是在顯示這個 binary 是否編入 CAN 控車能力。
  - `keypad` 不是泛指所有熱鍵是否能用，而是 `evdev` keypad reader 是否成功啟動。

### 14.3 思考與決策摘要（可公開版本）
- **決策 Q：CAN 接收與 CAN 控車 TX 明確拆分**
  - 原因：用戶要求預設安全模式，只收車況，不直接控車。
  - 做法：程式啟動後固定執行 `canbus_recv(CAN)`；是否送出控車訊號完全由 runtime toggle 決定。

- **決策 R：TX 採總開關 + 子開關雙層結構**
  - 原因：避免單一按鍵誤觸就同時打開油門煞車與方向盤。
  - 做法：`TX master` 只負責總允許權；縱向與方向盤各有獨立開關，兩者都需滿足才會真正發送。

- **決策 S：方向盤啟用必走 EPS 重新接管序列**
  - 原因：用戶明確指出方向盤有固定 init 過程，若未補送控制請求可能無法重新接回。
  - 做法：新增 `canbus_set_steering_tx_enabled(int enabled)`，在 `enabled=1` 時送 `0x02 -> 0x03 -> 0x03` 後才啟動 steering thread。

- **決策 T：繪圖開關分離，不用單一全域畫面開關**
  - 原因：用戶要求可獨立分析不同演算法輸出，不能把推論、ACC、behavior、collision 綁死。
  - 做法：runtime state 分成 `draw_inference_overlay`、`draw_acc_overlay`、`draw_behavior_overlay`、`draw_collision_overlay`、`draw_status_hud`。

- **決策 U：log 責任從 `main` 抽出，但不更動演算法核心**
  - 原因：用戶要求整理 `main`，同時不希望破壞既有研究紀錄與 ablation 能力。
  - 做法：新增 `RuntimeLogManager` 封裝 logger 啟動、virtual road 模擬入口、每幀打包與 stop 流程。

### 14.4 架構概要（本輪新增 / 調整）

#### 14.4.1 Keypad 執行期控制層
- `include/Keypad/keypad.h`, `src/keypad/keypad.cpp`
  - 功能：讀取 `evdev` keypad，並支援 OpenCV `waitKey()` fallback。
- `include/Keypad/keypad_control.h`, `src/keypad/keypad_control.cpp`
  - 功能：維護 runtime control state、熱鍵切換、HUD 狀態繪製、CAN runtime 同步。

#### 14.4.2 CAN 控制拆分層
- `CANBus/itri/include/canbus_recv.h`
- `CANBus/itri/src/canbus_recv.cpp`
  - 新增 `canbus_set_steering_tx_enabled(int enabled)`。
  - 啟用 steering TX 時會執行：
    - `send_V_Rq_EPS_Ctrl()`
    - `send_Rq_EPS_Ctrl()`
    - `send_Rq_EPS_Ctrl()`
  - 之後才進入 `canbus_ctrl_steer(1)`。

#### 14.4.3 Overlay / HUD 控制層
- 推論層：
  - `Engine/TFlite/include/TFlite_main.h`
  - `Engine/TFlite/src/TFlite_main.cc`
  - `Engine/TensorRT/include/TensorRT_main.hpp`
  - `Engine/TensorRT/src/TensorRT_main.cc`
  - 功能：支援 `draw_visuals` 開關。
- Behavior 層：
  - `src/VehicleBehavior/VehicleSkeletonProcessor.cpp`
  - 功能：依 draw params 決定是否畫 skeleton / heading。
- HUD：
  - `src/keypad/keypad_control.cpp`
  - 功能：統一顯示 TX、Throttle、Brake、Steering、overlay 狀態與熱鍵提示。

#### 14.4.4 Log 整理後的責任分層
- `include/log/research_data_logger.h`, `src/log/research_data_logger.cpp`
- `include/log/algorithm_ablation_logger.h`, `src/log/algorithm_ablation_logger.cpp`
- `include/log/time_sync.h`, `src/log/time_sync.cpp`
- `include/log/runtime_log_manager.h`, `src/log/runtime_log_manager.cpp`
  - `RuntimeLogManager` 負責：
    - 啟動 / 停止 research logger
    - 啟動 / 停止 ablation logger
    - `virtual_road` 模式下直接執行模擬
    - 每幀封裝 `FrameSnapshot` 後轉寫至 research / ablation logger
- `src/main.cpp`
  - 保留主流程與控制流程，不再內嵌大段 logger 初始化與 research/ablation CSV 封包細節。

### 14.5 熱鍵與執行期開關語意
- `1`：`TX master`
  - 只控制「是否允許送控車 CAN TX」。
  - 不會自動打開油門煞車或方向盤。
- `2`：`Speed/Brake`
  - 控制縱向輸出（油門 / 煞車）。
- `3`：`Steer`
  - 控制方向盤輸出。
- `4`：推論結果繪圖
- `5`：ACC overlay
- `6`：Behavior overlay
- `7`：Collision overlay
- `8`：HUD 顯示
- `0`：全部 overlay 一次切換
- `Backspace`：強制關閉所有控車輸出

補充語意：
- 真正縱向啟用條件：`TX master=ON` 且 `Speed/Brake=ON`
- 真正方向盤啟用條件：`TX master=ON` 且 `Steer=ON`
- 因此：
  - 只按 `1` 不會控車
  - `1 + 2` 才會啟用油門煞車
  - `1 + 3` 才會啟用方向盤
  - `1 + 2 + 3` 才會同時啟用縱向與方向盤

### 14.6 HUD 狀態欄位說明（本輪問答整理）
- `CAN compile`
  - 表示目前執行中的 binary 是否編入 `CANBUS__` 能力。
  - 不是表示 CAN 線上目前有沒有收到資料。
- `keypad`
  - 表示 `evdev` keypad reader 是否成功開啟。
  - 不是表示 OpenCV `waitKey()` fallback 一定不可用。
- `TX master`
  - 表示是否允許送出控車 CAN TX。
  - 不是表示是否接收 CAN。
- `Throttle / Brake / Steering`
  - 顯示目前各控制輸出是否處於有效啟用條件。

### 14.7 本輪主要改動檔案
- Keypad / runtime control：
  - `include/Keypad/keypad.h`
  - `src/keypad/keypad.cpp`
  - `include/Keypad/keypad_control.h`
  - `src/keypad/keypad_control.cpp`
- CAN steering re-enable：
  - `CANBus/itri/include/canbus_recv.h`
  - `CANBus/itri/src/canbus_recv.cpp`
- Runtime config：
  - `include/system_config.h`
  - `src/system_config.cpp`
  - `config/system_config.yaml`
- Overlay draw control：
  - `Engine/TFlite/include/TFlite_main.h`
  - `Engine/TFlite/src/TFlite_main.cc`
  - `Engine/TensorRT/include/TensorRT_main.hpp`
  - `Engine/TensorRT/src/TensorRT_main.cc`
  - `src/VehicleBehavior/VehicleSkeletonProcessor.cpp`
- Log 分層整理：
  - `include/log/research_data_logger.h`
  - `src/log/research_data_logger.cpp`
  - `include/log/algorithm_ablation_logger.h`
  - `src/log/algorithm_ablation_logger.cpp`
  - `include/log/time_sync.h`
  - `src/log/time_sync.cpp`
  - `include/log/runtime_log_manager.h`
  - `src/log/runtime_log_manager.cpp`
  - `src/main.cpp`
  - `CMakeLists.txt`

### 14.8 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
- 結果：
  - 預設行為已改為 receive-only CAN。
  - 執行期可分開切換總 TX、縱向、方向盤。
  - 方向盤重新啟用時會先做 EPS 初始化序列。
  - 推論/ACC/Behavior/Collision/HUD 可獨立開關。
  - `main` 的 logger 細節已抽離到 `src/log` / `include/log`。

### 14.9 已知限制與風險（本輪）
- `CAN compile` 與 `keypad` 目前用語偏底層，容易被誤解為「是否有 CAN 訊號」或「熱鍵完全失效」；若後續要降低誤判，可再調整 HUD 文案。
- `keypad` 欄位只反映 `evdev` reader 成功與否；當它為 `OFF` 時，若 OpenCV 視窗仍有焦點，`waitKey()` 路徑可能仍可收到按鍵。
- HUD 位置目前仍寫死在 `DrawRuntimeStatusOverlay()` 中；若未來需要不同解析度自適應，建議再配置化。

### 14.10 後續建議（本輪）
1. 若實車操作人員較多，建議把 HUD 文案改成更直白的術語，例如 `CAN TX armed`、`evdev reader`。
2. 可進一步把 HUD 座標、字體大小、行距加入 `system_config.yaml`，避免每次都改碼。
3. 若未來還要擴充 runtime control，可把更多診斷資訊（例如最近 steering init 是否成功）也掛進 HUD。

---

## 15. 本輪再追加（鏡頭重架後的 IPM / 單應性對應重整）

### 15.1 本輪用戶要求（分類）
1. 因鏡頭重新架設，需依新座標對應重新調整 IPM / 單應性相關參數。
2. 新的鏡頭座標與影像座標已提供於 `./Camera-Config/單應性1280x720.csv`。
3. 需將可被系統實際使用的結果更新進 `./Camera-Config/Sensing-3M.yaml`。
4. 需同步把本輪對話的重點、需求、決策與結果整理回 `./AGENT.md`。

### 15.2 歷史對話重點（本輪濃縮）
- 用戶先說明鏡頭已重新安裝，因此舊 IPM / 地面投影參數不再可信。
- 提供新的 1280x720 地面平面對應點 CSV，內容包含像素座標與地面深度/橫向座標。
- 先檢查 `Sensing-3M.yaml` 現況，確認檔內實際存的是 `K`、`D`、`R_cw`、`t_cw`，而不是獨立 `H` 單應性欄位。
- 再追查專案讀取流程，確認 `CameraModel` 與 `WorldProjector` 真正使用的是外參 `R_cw/t_cw` 進行地面投影。
- 校正過程中發現本專案 raw world 軸序與 CSV 欄名語意不完全相同；需先對齊軸定義，否則會把前進/橫向搞反。

### 15.3 思考與決策摘要（可公開版本）
- **決策 V：不新增 YAML 單應性欄位，直接更新系統真正在用的外參**
  - 原因：程式碼讀的是 `K/D/R_cw/t_cw`；若只寫一組新的 homography 到其他欄位，主流程不會使用。
  - 做法：保留 `K` 與 `D`，依新 CSV 對應點重解並覆寫 `R_cw`、`t_cw`。

- **決策 W：先對齊專案座標軸，再做外參重解**
  - 原因：`單應性1280x720.csv` 欄名是「深度 / 橫向」，但專案 geometry raw world 實際採 `X=橫向, Y=前進, Z=0`。
  - 做法：在解算與文件中明確採用 `X=lateral_cm, Y=forward_cm, Z=0`，避免軸交換造成嚴重投影偏差。

- **決策 X：保留原鏡頭內參與畸變，只更新重架後的姿態**
  - 原因：本輪輸入只有地面平面對應點，較適合修正鏡頭外參；若連 `K/D` 一起重估，風險較高。
  - 做法：沿用既有 `K`、`D`，把 `R_cw/t_cw` 視為鏡頭重架後需更新的主體。

- **決策 Y：以 CSV 重投影誤差作為本輪驗證基準**
  - 原因：用戶提供的是地面對應點，因此最直接的驗證就是檢查更新後 YAML 對這組點的像素回投影誤差。
  - 做法：用更新後的 `Sensing-3M.yaml` 回算整份 CSV，確認 RMSE 明顯優於修改前設定。

### 15.4 架構 / 設定層影響（本輪新增 / 調整）

#### 15.4.1 Camera 標定設定層
- `Camera-Config/Sensing-3M.yaml`
  - 保留：
    - `K`
    - `D`
  - 更新：
    - `R_cw`
    - `t_cw`
  - 補充註解：
    - 說明本專案 raw world 座標軸為 `X=lateral_cm, Y=forward_cm, Z=0`
    - 註明此次更新來源為 `單應性1280x720.csv`

#### 15.4.2 幾何投影消費層
- `include/Geometry/CameraModel.h`, `src/Geometry/CameraModel.cpp`
  - 持續從 YAML 載入 `K/D/R_cw/t_cw`。
- `src/Geometry/WorldProjector.cpp`
  - 持續以 `Rwc = Rcw^T` 與 `cameraCenterWorld()` 對地面 `Z=0` 做 ray-plane intersection。
- `src/Geometry/GeometryFunction.cpp`
  - 持續把 raw world 轉成系統內部車體座標語意：`x_forward_m`、`y_left_m`。

### 15.5 本輪主要改動摘要（檔案分組）

#### 15.5.1 既有檔案調整
- `Camera-Config/Sensing-3M.yaml`
  - 更新重架鏡頭後的 `R_cw` / `t_cw`
  - 補充世界座標軸註解
- `AGENT.md`
  - 追加本輪鏡頭/IPM 校正整理

#### 15.5.2 本輪參考輸入
- `Camera-Config/單應性1280x720.csv`
  - 作為新的地面平面像素/世界對應資料來源

### 15.6 驗證與結果（本輪）
- 驗證方式：
  - 以更新後 `Sensing-3M.yaml` 的 `R_cw/t_cw` 回投影 `單應性1280x720.csv` 的所有點。
- 驗證結果：
  - 更新前，舊 YAML 對這組新點的重投影 RMSE 約為 `252.60 px`。
  - 更新後，新的 YAML 對同組點的重投影 RMSE 約為 `65.94 px`。
- 結論：
  - 新鏡頭姿態已成功反映到 `Sensing-3M.yaml`。
  - 以本輪提供的地面對應點來看，更新後結果明顯優於修改前設定，可作為新的基準外參使用。

### 15.7 已知限制與風險（本輪）
- 本輪調整的是地面平面投影對應，主體是外參修正；`K/D` 並未重新完整標定。
- 遠距離點（特別是約 40m 以上）殘差仍偏大，表示目前 CSV 本身可能帶有量測誤差、視差誤差或遠距標定不穩定現象。
- 若後續主要使用情境集中在近中距離車道 / 目標投影，建議以 3m 到 20m 區間做加權重解，可能比全距離平均更符合實車需求。
- `Sensing-3M.yaml` 目前仍是以外參形式描述 IPM 能力；若未來想顯式保存 `image->ground` 或 `ground->image` homography，可再另外擴充格式，但需同步改讀取流程。

### 15.8 後續建議（本輪）
1. 若實際控制與車道分析主要看近中距離，建議新增「近距優先」版本標定，降低遠距 noisy 點對結果的影響。
2. 建議補拍一組新的驗證影像，人工檢查 lane / ground point 回投影是否與實際路面對齊。
3. 若後續鏡頭角度還會再微調，建議把 CSV -> 外參更新流程整理成固定腳本，避免每次人工重算。


---

## 16. 本輪再追加（LKA 參考點可視化與 research log 透明化）

### 16.1 本輪用戶要求（分類）
1. 根據 LKA 實際控制邏輯，將「車輛當前點」與「欲前往的目標點」直接畫在影像上。
2. 該繪圖需可透過 keypad 在執行期開關，不能寫死在畫面上。
3. 將當前點與目標點同步寫入 log，讓使用者能在當下與後處理時理解 LKA 的運行依據，降低黑盒感。

### 16.2 思考與決策摘要（可公開版本）
- **決策 Z：直接沿用 LKA 控制內部的 reference snapshot，不另建第二套幾何推估**
  - 原因：若繪圖點與控制點不是同一份資料，畫面與 log 會變成「看起來像」而不是「實際上就是」。
  - 做法：在 `lane_keeping` 內保存每幀最新的 `LkaReferenceSnapshot`，由 `main` 讀出後投影到影像與寫入 research log。

- **決策 AA：將 LKA 點位繪圖做成獨立 runtime overlay**
  - 原因：用戶要求能單獨開關，且不應綁在 inference / behavior / collision 其中之一。
  - 做法：新增 `draw_lka_overlay` runtime state，並用 keypad 熱鍵獨立切換。

- **決策 AB：同時記錄世界座標與影像座標**
  - 原因：離線分析時只記 px 不足以理解控制幾何，只記 meter 又無法直接對照畫面。
  - 做法：research CSV 同步保存 `x/y (m)` 與 `u/v (px)`，並附上 valid flag。

### 16.3 架構概要（本輪新增 / 調整）
- `include/LKA/lane_keeping.h`, `src/LKA/lane_keeping.cpp`, `src/LKA/lk_stanley_controller.cpp`
  - 新增 `LkaReferenceSnapshot`。
  - 每幀保存 LKA 控制正在使用的 current / target reference point。
- `src/main.cpp`
  - 將 LKA reference point 投影回影像並繪製。
  - 把 current / target 的 meter 與 pixel 資訊封裝進 `FrameSnapshot`。
- `include/Keypad/keypad_control.h`, `src/keypad/keypad_control.cpp`
  - 新增 `draw_lka_overlay`。
  - 熱鍵改為：`6=LKA`, `7=Behavior`, `8=Collision`, `9=HUD`。
- `include/log/runtime_log_manager.h`, `src/log/runtime_log_manager.cpp`
  - 新增 LKA reference 點位欄位傳遞。
- `include/log/research_data_logger.h`, `src/log/research_data_logger.cpp`
  - research CSV 新增：
    - `lka_reference_valid`
    - `lka_p_curve`
    - `lka_current_x_m / lka_current_y_m / lka_current_u_px / lka_current_v_px`
    - `lka_target_x_m / lka_target_y_m / lka_target_u_px / lka_target_v_px`
- `include/system_config.h`, `src/system_config.cpp`, `config/system_config.yaml`
  - 新增 `draw_lka_overlay` 初始配置。

### 16.4 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
- 結果：
  - 畫面可顯示 LKA current / target point。
  - 可用 keypad 在執行期切換 LKA overlay。
  - research log 已可回看當幀 LKA 的參考點與投影位置。

### 16.5 已知限制與風險（本輪）
- `current point` 採用的是 LKA 控制 reference point（ego anchor），`target point` 採用的是控制預覽目標點；這是為了對齊控制邏輯，不是單純畫車頭中心。
- 當影像投影點落在畫面外或當幀無有效 lane 時，CSV 仍會保留 valid flag 供離線判讀。
- 目前 research log 是逐幀點位紀錄，尚未額外輸出 LKA 點位的專用 summary / plot。

### 16.6 後續建議（本輪）
1. 若後續要更完整解釋 LKA，可再增加「當前 path 點」與「preview path 點」的分離欄位。
2. 可追加離線腳本，直接把 `lka_current_* / lka_target_*` 疊回影片或生成 point trajectory 圖。
3. 若實車操作人員需要更直觀控制，建議把 HUD 內 hotkey 說明同步更新到 README。


---

## 17. 本輪再追加（ACC 候選 / lead / 真正跟車 lead 的分色可視化）

### 17.1 本輪用戶要求（分類）
1. 將 ACC overlay 從單純的「target / 非 target」二分，提升為多狀態顯示。
2. 需分別標示：候選車、lead、真正進入跟車的 lead、與剩餘車輛。
3. 顏色語意需盡量對齊實際 ACC 控制狀態，避免只是在畫面上重算一套近似規則。

### 17.2 思考與決策摘要（可公開版本）
- **決策 AC：ACC overlay 狀態直接由 `AccCommand` 攜帶，不在 draw 階段重跑第二套判斷**
  - 原因：若畫面端自己再做 heuristic，容易和實際控制 lead 狀態脫鉤。
  - 做法：在 `AccCommand` 內新增 `candidate_ids` 與 `lead_following_active`，由 ACC controller 在同一幀控制計算後直接輸出。

- **決策 AD：將「真正進入跟車」定義為前車已實際壓低縱向命令，而非單純被選成 lead**
  - 原因：被選為 lead 不代表此刻已經進入明顯跟車；遠距離目標可能仍幾乎等同自由巡航。
  - 做法：以「與 free-road 加速度相比，lead 是否造成足夠顯著的縱向命令下降」作為 `lead_following_active` 判定基準。

### 17.3 架構概要（本輪新增 / 調整）
- `include/ACC/AccConfig.h`
  - `AccCommand` 新增：
    - `candidate_ids`
    - `lead_following_active`
- `include/ACC/AccController.h`
  - ACC controller 每幀輸出候選清單。
  - 新增 `lead_following_active` 判定，表示 lead 已實際限制縱向控制。
- `include/ACC/AccDebugDraw.h`
  - overlay 顏色更新為四類：
    - `FOLLOW`：真正進入跟車的 lead
    - `LEAD`：已選為 lead，但尚未明顯進入跟車限制
    - `CAND`：候選車
    - `REM`：剩餘車輛

### 17.4 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
- 結果：
  - ACC overlay 不再只有綠 / 紅兩種語意。
  - 畫面可直接區分「只是候選」與「真正已經影響 ACC 跟車控制」的前車。

### 17.5 已知限制與風險（本輪）
- `lead_following_active` 目前定義為「lead 已對縱向命令造成足夠顯著影響」；這是控制語意，不是法律或論文標準定義。
- overlay 仍只顯示 `class_id` 為 1/2/3 的動態目標，不會把其他類別納入 ACC 顏色分級。
- `candidate_ids` 是本幀結果，未額外做視覺平滑；快速切換場景時顏色仍可能隨感知結果跳動。

### 17.6 後續建議（本輪）
1. 若後續要做更論文化分析，可把 ACC overlay 狀態同步寫進 research log。
2. 可追加畫面 legend，直接在 HUD 顯示各顏色代表的 ACC 狀態。
3. 若要更保守，可把 `lead_following_active` 的門檻配置化，方便實車調整語意敏感度。

## 18. 本輪再追加（ACC 四種車輛狀態落盤 + 縱向 phase HUD / log）
- 把 ACC 四種視覺狀態 `candidate / lead / following_lead / remaining` 寫進 research log，並加入每幀 `acc_object_state_summary`，摘要包含 `id / class / score / ground(x,y) / image(u,v) / box(x,y,w,h)`。
- 追加 lead 驗證欄位：`acc_target_id`、`acc_target_distance_m`、`acc_target_lateral_m`、`acc_target_relative_speed_mps`、`acc_target_score`、`acc_target_dist_std_m`、`acc_target_rel_speed_std_mps`、`acc_target_box_*`、`acc_target_bottom_center_*`。
- 追加 ACC 縱向控制 phase：`max_hold / accelerating / idle / braking`，同時輸出 `acc_control_ego_speed_kmh`、`acc_control_cruise_speed_kmh`、`acc_control_speed_cmd_kmh`、`acc_control_brake_0_10`、`acc_control_accel_cmd_mps2`、`acc_control_free_accel_*`。
- ACC overlay 右上角新增 `MAX HOLD / ACCEL / IDLE / BRAKE` phase HUD，僅高亮當前 phase，畫面與 log 共用同一份 `AccCommand.longitudinal_phase` 判斷，避免畫面與 CSV 語意漂移。

## 19. 本輪再追加（FPS / 每功能 ms 右下角 HUD + log）
- 新增主迴圈 performance metrics：`fps`、`total_ms`、`input_ms`、`inference_ms`、`geometry_ms`、`behavior_ms`、`collision_ms`、`overlay_ms`。
- 新增 VehicleControl 內部分項：`acc_scope_ms`、`acc_ms`、`lka_ms`、`stability_ms`、`control_total_ms`。
- HUD 模式下在畫面右下角顯示 performance panel，標示各功能耗時與即時 FPS / total frame time。
- research log 追加對應 perf 欄位，方便事後比對哪一段變慢。

## 20. 本輪再追加（地面世界座標 1m 網格 overlay 與 keypad 切換）

### 20.1 本輪用戶要求（分類）
1. 依 `./Camera-Config/Sensing-3M.yaml` 的內參 / 外參，把地面平面世界座標網格直接投影回影像。
2. 需顯示縱向 / 橫向每公尺的對應網格線，方便現場確認單應性 / ground projection 是否合理。
3. 該網格必須能透過 keypad 在執行期開關，不能寫死。

### 20.2 思考與決策摘要（可公開版本）
- **決策 AE：網格直接使用系統實際載入的 `CameraModel` 投影**
  - 原因：若另外重算一套簡化 homography，畫面看到的網格不一定等於系統實際使用的相機模型。
  - 做法：以 `Sensing-3M.yaml` 載入的 `K/D/R_cw/t_cw` 為唯一來源，將地面世界點投影回影像。

- **決策 AF：補上「標定解析度 -> 當前影像解析度」縮放投影**
  - 原因：目前執行期畫面可能是 1280x720，但相機 YAML 仍保存 1920x1080 標定尺寸；若直接用原始 `K` 投影，overlay 可能整體縮放錯位。
  - 做法：`CameraModel` 額外保存 `image_width/image_height`，投影後依當前影像尺寸做等比例縮放。

- **決策 AG：地面網格做成獨立 runtime overlay，不綁在 LKA / Behavior / Collision**
  - 原因：使用者要拿它做標定/實驗確認，語意上是相機幾何檢查工具，不應綁進其他演算法圖層。
  - 做法：新增獨立 `draw_ground_grid_overlay` 狀態與 `G` 熱鍵切換。

### 20.3 架構概要（本輪新增 / 調整）
- `include/Geometry/CameraProjectionUtils.h`, `src/Geometry/CameraProjectionUtils.cpp`
  - 功能：統一處理 vehicle meter -> raw world cm -> image pixel 的投影，並自動處理標定尺寸與當前畫面尺寸的縮放。
- `include/Geometry/WorldGridOverlay.h`, `src/Geometry/WorldGridOverlay.cpp`
  - 功能：把地面 `forward / lateral` 每 1m 網格線投影到影像，並以不同顏色畫 major/minor line。
- `include/Geometry/CameraModel.h`, `src/Geometry/CameraModel.cpp`
  - 功能：保存 `image_width / image_height`，供縮放投影使用。
- `include/Keypad/keypad_control.h`, `src/keypad/keypad_control.cpp`
  - 功能：新增 `draw_ground_grid_overlay` runtime state 與 `G` 熱鍵切換。
- `src/main.cpp`
  - 功能：在主流程 overlay 階段呼叫 `DrawWorldGridOverlay(...)`。
- `src/LKA/lk_visualization.cpp`
  - 功能：LKA 視覺化也改走同一套縮放投影 helper，避免與 ground grid 使用不同座標映射。

### 20.4 本輪主要改動檔案
- Geometry / overlay：
  - `include/Geometry/CameraProjectionUtils.h`
  - `src/Geometry/CameraProjectionUtils.cpp`
  - `include/Geometry/WorldGridOverlay.h`
  - `src/Geometry/WorldGridOverlay.cpp`
  - `include/Geometry/CameraModel.h`
  - `src/Geometry/CameraModel.cpp`
- Runtime control / main：
  - `include/Keypad/keypad_control.h`
  - `src/keypad/keypad_control.cpp`
  - `include/system_config.h`
  - `src/system_config.cpp`
  - `config/system_config.yaml`
  - `src/main.cpp`
- 對齊投影邏輯：
  - `src/LKA/lk_visualization.cpp`

### 20.5 使用方式（本輪）
1. 預設可透過 `app.draw_ground_grid_overlay` 控制啟動時是否顯示。
2. 執行期可按 `G` 切換 world grid overlay。
3. `app.ground_grid_*` 可調整：
   - 前向範圍
   - 左右範圍
   - 網格間距
   - 採樣步距
   - major line 間隔
   - 是否顯示文字標籤

### 20.6 驗證與結果（本輪）
- 建置驗證：
  - `cmake -S . -B build-TFlite`：通過
  - `cmake --build build-TFlite -j4`：通過
  - `cmake -S . -B build-TensorRT`：通過
  - `cmake --build build-TensorRT -j4`：通過
- 結果：
  - 影像可顯示依 `Sensing-3M.yaml` 投影的地面世界座標網格。
  - 可用 `G` 熱鍵即時開關。
  - 投影 helper 已考慮標定解析度與當前畫面解析度不同的情況。

### 20.7 已知限制與風險（本輪）
- 網格畫的是地面 `Z=0` 平面；若實際路面有坡度、起伏或攝影機姿態再變動，仍會出現局部偏差。
- 現在的 major/minor 標籤偏向人工檢視用途，還不是完整標定診斷工具。
- 若輸入影像又被額外裁切或非等比例縮放，仍需重新檢查相機模型與實際顯示畫面的對應關係。

### 20.8 後續建議（本輪）
1. 若要做更嚴格驗證，可再加「滑鼠點像素 -> 反投影世界座標」的互動診斷工具。
2. 可把 grid overlay 的顏色、字體大小、標籤密度再配置化。
3. 若後續要做論文化附圖，可加上 ego origin、車體寬度與 lane width reference 線。

## 21. 本輪再追加（LKA 左右車道線 / 中線 0~20m overlay + ego / reference point）

### 21.1 本輪用戶要求（分類）
1. 在現有 `draw_lka_overlay` 路徑中，加入 LKA 實際計算出的左右車道線與中線。
2. 上述線條需以世界座標 `0m -> 20m` 的前向範圍投影回影像。
3. 保留既有的 LKA current / target point，並額外畫出當前車輛點（ego origin）。

### 21.2 思考與決策摘要（可公開版本）
- **決策 AH：沿用 LKA 現有 lane selection / centerline 邏輯，不在主畫面重建另一套規則**
  - 原因：若畫面上的左右線 / 中線和控制實際使用的資料不是同一條鏈，會再次產生黑盒感。
  - 做法：直接重用 `FindBestLaneCandidates(...)` 與 `BuildCenterlineFromWorldResult(...)`。

- **決策 AI：左右線與中線以固定 `0~20m` 世界座標區間可視化**
  - 原因：用戶明確要看從車前 `0m` 到 `20m` 的幾何線形，而不是只看目前可用點的重疊區間。
  - 做法：visualization helper 將擬合曲線投影到固定前向範圍；若缺少 polynomial，則回退到可用點區間內的線性插值。

- **決策 AJ：ego origin 和 current / target point 同時顯示**
  - 原因：使用者要同時確認「車輛當前位置」、「LKA 控制 reference current point」與「preview target point」三者相對關係。
  - 做法：保留既有 `current/target`，並新增 `ego` 點位與文字標示。

### 21.3 架構概要（本輪新增 / 調整）
- `include/LKA/lk_visualization.h`, `src/LKA/lk_visualization.cpp`
  - 新增 `DrawLkaLaneSolutionOnImage(...)`。
  - 功能：繪製：
    - 左車道線
    - 右車道線
    - 中線
  - 前向範圍固定為 `0~20m`。
- `src/main.cpp`
  - 在 `draw_lka_overlay` 路徑中呼叫 `DrawLkaLaneSolutionOnImage(...)`。
  - `DrawLkaReferenceOverlay(...)` 新增 `ego` 點位繪製。

### 21.4 本輪主要改動檔案
- `include/LKA/lk_visualization.h`
- `src/LKA/lk_visualization.cpp`
- `src/main.cpp`

### 21.5 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
  - `cmake --build build-TensorRT -j4`：通過
- 結果：
  - 啟用 `draw_lka_overlay` 時，畫面可同時顯示：
    - 左右車道線
    - 中線
    - ego origin
    - LKA current point
    - LKA target point

### 21.6 已知限制與風險（本輪）
- 固定畫到 `20m` 代表在某些幀會有外插；若當幀有效 lane 點只集中在近距或中距，遠端線段可信度會較低。
- 當左右車道 polynomial 不可用時，會退回線性插值，因此畫面上不同幀的線條語意可能在「quadratic fit」與「linear fallback」之間切換。
- ego origin 若投影落在畫面外，該點不會出現；這取決於相機姿態與畫面裁切範圍。

### 21.7 後續建議（本輪）
1. 若要讓使用者更容易判讀，可再加 legend，標明 `L / R / Center / Ego / Current / Target` 顏色語意。
2. 若後續要與論文圖一致，可把 `0~20m` 改成 YAML 配置而非寫死。
3. 若想區分「控制實際使用的中心線」與「純視覺化的外插線」，可再額外疊加 observed-range 標記。

## 22. 本輪再追加（Lane detect 偏移 / 壓線世界座標檢測 + H 熱鍵）

### 22.1 本輪用戶要求（分類）
1. 新增一項獨立的 Lane detect 功能與 keypad 按鍵。
2. 偏移/壓線判斷需直接使用世界座標，不要透過二次曲線公式做判定。
3. 平常先把原始車道線畫成綠色。
4. 當變換車道或車道線碰到車輛邊緣時，將對應車道線改成紅色，並在畫面下方額外加粗標示那一條偏移的車道線。

### 22.2 思考與決策摘要（可公開版本）
- **決策 AK：Lane detect 與 LKA curve fit 分離**
  - 原因：用戶明確要求「不要透過二次曲線公式轉成曲線判斷」。
  - 做法：另外建立 direct world-coordinate lane detection 路徑，只使用：
    - `ExtractLanePointsVehicleM(...)`
    - `EstimateLaneYAtX(...)`
    - `SampleYLinear(...)`

- **決策 AL：壓線判定採近距離車道線與車身半寬比較**
  - 原因：用戶要的是「車道線碰到車輛邊緣」與變換車道時的即時判斷。
  - 做法：在近距離前向範圍內，以左/右車道線的世界座標 `y` 與 `vehicle_half_width_m` 比較；若左線進入左車身邊界或右線進入右車身邊界，即視為偏移/壓線。

- **決策 AM：新增獨立 `H` 熱鍵與 overlay state**
  - 原因：此功能與 LKA overlay、grid overlay 語意不同，需可單獨開關。
  - 做法：新增 `draw_lane_detect_overlay` 與 `H:LaneDet`。

### 22.3 架構概要（本輪新增 / 調整）
- `include/LKA/lk_visualization.h`, `src/LKA/lk_visualization.cpp`
  - 新增 `DrawLaneDetectOverlayOnImage(...)`。
  - 功能：
    - 用原始世界座標線性插值選左右車道
    - 畫綠色原始車道線
    - 壓線/偏移時改畫紅色
    - 在畫面下方額外加粗標示偏移那一條線
- `include/LKA/lane_keeping.h`
  - 新增 lane detect 參數：
    - `lane_detect_vehicle_half_width_m`
    - `lane_detect_forward_range_m`
    - `lane_detect_draw_end_m`
    - `lane_detect_bottom_range_m`
    - `lane_detect_contact_margin_m`
- `include/Keypad/keypad_control.h`, `src/keypad/keypad_control.cpp`
  - 新增 `draw_lane_detect_overlay`
  - 新增 `H` 熱鍵
  - HUD 顯示 `LaneDet`
- `include/system_config.h`, `src/system_config.cpp`, `config/system_config.yaml`
  - 新增 app-level `draw_lane_detect_overlay`
  - 新增 lka-level lane detect 參數
- `src/main.cpp`
  - 在主畫面 overlay 階段呼叫 `DrawLaneDetectOverlayOnImage(...)`

### 22.4 使用方式（本輪）
1. `H`：開關 `Lane detect overlay`
2. 預設啟動由 `app.draw_lane_detect_overlay` 控制
3. 以下參數可在 `lka:` 區塊調整：
   - `lane_detect_vehicle_half_width_m`
   - `lane_detect_forward_range_m`
   - `lane_detect_draw_end_m`
   - `lane_detect_bottom_range_m`
   - `lane_detect_contact_margin_m`

### 22.5 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
  - `cmake --build build-TensorRT -j4`：通過
- 結果：
  - 平時左右車道線以綠色顯示。
  - 當左/右車道線進入車身邊界判定時，對應線條改為紅色。
  - 偏移那一條線會在畫面下方額外加粗標示。
  - 可透過 `H` 即時開關。

### 22.6 已知限制與風險（本輪）
- 此功能刻意不使用 quadratic fit，所以遠距離或點稀疏時，線性插值顯示會較保守。
- 左右車道候選仍需先有足夠的原始 lane world points；若某一側點太少，該側不會畫也不會判斷。
- 若畫面裁切導致近距離 lane points 缺失，壓線偵測可能變得延後或不穩定。

### 22.7 後續建議（本輪）
1. 若要更像車道偏移警示系統，可再加入時間去抖動與連續幀 hysteresis。
2. 可把左/右壓線狀態同步寫入 research log，方便事後統計偏移事件。
3. 若實車車寬與感知原點不同，建議再實測校正 `lane_detect_vehicle_half_width_m`。

## 23. 本輪再追加（Lane detect 改為 YAML-only 模式切換：direct_world / quadratic_curve）

### 23.1 本輪用戶要求（分類）
1. Lane detect 不再使用 keypad 控制。
2. 改為以 YAML 變數決定是否啟用與判定模式。
3. 需支援兩種模式：
   - `direct_world`
   - `quadratic_curve`

### 23.2 思考與決策摘要（可公開版本）
- **決策 AN：Lane detect 顯示開關改回靜態配置，不走 runtime hotkey**
  - 原因：用戶明確要求不用 keypad 控制。
  - 做法：保留 `app.draw_lane_detect_overlay` 作為 YAML-only 開關，移除 `H` 熱鍵對 Lane detect 的切換語意。

- **決策 AO：把「判定來源」抽成 mode，而不是再加第二套 overlay**
  - 原因：使用者要比較的是同一個 Lane detect 視覺功能，但其偏移判定依據不同。
  - 做法：新增 `lka.lane_detect_mode`，支援：
    - `direct_world`：原始世界座標點 + 線性插值
    - `quadratic_curve`：左右車道二次擬合後再做偏移判定

- **決策 AP：模式切換只影響偏移判定與 lane sampling，不改紅/綠視覺語意**
  - 原因：避免模式改變時，操作者還要重新學一套顏色規則。
  - 做法：兩種模式都維持：
    - 正常：綠線
    - 偏移/壓線：紅線
    - 下方加粗標示偏移那一側

### 23.3 架構概要（本輪新增 / 調整）
- `include/LKA/lane_keeping.h`
  - 新增 `lane_detect_mode`
- `src/LKA/lk_visualization.cpp`
  - 新增 mode parsing 與 `BuildLaneDetectPair(...)`
  - `direct_world`：走 raw points + `SampleYLinear(...)`
  - `quadratic_curve`：走 `FindBestLaneCandidates(...)` 的 quadratic fit 結果
- `src/keypad/keypad_control.cpp`
  - 移除 Lane detect 的 runtime hotkey 處理與 HUD 顯示
- `src/main.cpp`
  - Lane detect 是否繪製改由 `runtime_cfg.app.draw_lane_detect_overlay` 直接控制
- `src/system_config.cpp`, `config/system_config.yaml`
  - 新增 `lka.lane_detect_mode`

### 23.4 使用方式（本輪）
1. 是否顯示 Lane detect：
   - `app.draw_lane_detect_overlay: 0|1`
2. 判定模式：
   - `lka.lane_detect_mode: "direct_world"`
   - `lka.lane_detect_mode: "quadratic_curve"`

### 23.5 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
  - `cmake --build build-TensorRT -j4`：通過
- 結果：
  - Lane detect 已改為 YAML-only 控制。
  - `H` 不再切換 Lane detect。
  - 可用 YAML 在 direct world 與 quadratic curve 間切換。

### 23.6 已知限制與風險（本輪）
- `quadratic_curve` 模式在 lane polynomial 不可用時，仍會退回原始點線性插值，避免當幀完全失效。
- 若後續有人只看舊版對話紀錄，可能還會以為 `H` 能切 Lane detect；以本節為最新準則。

### 23.7 後續建議（本輪）
1. 若要做模式比較，建議把目前生效的 `lane_detect_mode` 也畫到 HUD 或寫進 log。
2. 若後續要完全禁止 Lane detect 與 keypad 互動，可連 `CMD_H` 在 legacy keypad 路徑中的歷史映射一起清掉。

## 24. 本輪再修正（Lane detect 改為 keypad 控制顯示，YAML 僅控制判定模式）

### 24.1 本輪用戶要求（分類）
1. Lane detect 的繪圖顯示要恢復成 keypad 控制。
2. 左右車道線偏移判斷方式仍由 YAML 決定。
3. 需保留兩種判定模式：
   - `direct_world`
   - `quadratic_curve`

### 24.2 思考與決策摘要（可公開版本）
- **決策 AQ：把 Lane detect 拆成「顯示開關」與「判定模式」兩層控制**
  - 原因：使用者要的是 runtime 視覺切換方便，但研究判定邏輯仍需固定可重現。
  - 做法：
    - 顯示開關：恢復為 keypad `H`
    - 判定模式：維持 `lka.lane_detect_mode`

- **決策 AR：保留 `app.draw_lane_detect_overlay` 作為啟動預設值**
  - 原因：讓系統開機時可預設顯示或不顯示，但執行期間仍可立即用 keypad 覆寫。
  - 做法：初始化 `RuntimeControlState.draw_lane_detect_overlay` 時沿用 YAML 值，之後由 `H` 切換。

### 24.3 架構概要（本輪新增 / 調整）
- `include/Keypad/keypad_control.h`
  - 恢復 `draw_lane_detect_overlay` runtime state
- `src/keypad/keypad_control.cpp`
  - 初始化 lane detect overlay 狀態
  - 恢復 `CMD_H` 切換
  - `CMD_0` 全 overlay 開關重新納入 Lane detect
  - HUD 熱鍵與狀態列恢復顯示 `LaneDet`
- `src/main.cpp`
  - Lane detect 是否繪製改回讀 `control_state.draw_lane_detect_overlay`
- `config/system_config.yaml`
  - `app.draw_lane_detect_overlay` 改註解為「啟動預設值」

### 24.4 使用方式（本輪）
1. 顯示開關：
   - 啟動預設值：`app.draw_lane_detect_overlay`
   - 執行期切換：keypad `H`
2. 判定模式：
   - `lka.lane_detect_mode: "direct_world"`
   - `lka.lane_detect_mode: "quadratic_curve"`

### 24.5 驗證與結果（本輪）
- 建置驗證：
  - `cmake --build build-TFlite -j4`：通過
  - `cmake --build build-TensorRT -j4`：通過
- 結果：
  - Lane detect overlay 可由 `H` 即時開關。
  - `app.draw_lane_detect_overlay` 現在作為啟動預設值，進入執行期後由 keypad 接手。
  - 偏移判定仍由 YAML 的 `lane_detect_mode` 決定，不受 keypad 影響。
