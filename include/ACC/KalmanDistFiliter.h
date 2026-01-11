#pragma once

#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>

namespace acc {

class AccKalmanAdapter {
public:
  AccKalmanAdapter() = default;

  /**
   * 設定濾波器參數 (Tuning)
   * @param meas_std_m: 距離量測標準差 [公尺]。
   * 數值越小，Filter 越相信量測值（反應快但容易抖動）；
   * 數值越大，Filter 越相信預測模型（平滑但會有延遲）。
   * @param accel_std_mps2: 過程噪聲（相對加速度）標準差 [m/s^2]。
   * 代表我們預期前車可能會有多大的加速度變化。
   * 數值越大，Q 矩陣越大，允許更劇烈的加減速（跟得緊），但也更抖。
   * @param gate_nis_soft: NIS 軟門檻（默認 9.0，約等於 3-sigma）。
   * 低於此值視為「正常更新」。
   * @param gate_nis_hard: NIS 硬門檻（默認 25.0，約等於 5-sigma）。
   * 介於 soft 與 hard 之間會進行「軟更新」（降低權重）。
   * 超過此值則「拒絕更新」（視為雜訊）。
   */
  void SetTuning(float meas_std_m,
                 float accel_std_mps2,
                 float gate_nis_soft = 9.0f,
                 float gate_nis_hard = 25.0f) {
    // 加上 std::max 進行防呆，避免參數為負或過小導致矩陣奇異
    meas_std_m_        = std::max(1e-3f, meas_std_m);
    accel_std_mps2_    = std::max(1e-3f, accel_std_mps2);
    gate_nis_soft_     = std::max(0.0f, gate_nis_soft);
    gate_nis_hard_     = std::max(gate_nis_soft_, gate_nis_hard);
  }

  /**
   * 主要更新函式：輸入新的測量距離，更新濾波器狀態
   * @param z_meas_dist: 當前偵測到的距離 [公尺]
   * @param dt: 與上一幀的時間差 [秒]
   * @param frame: 當前幀號
   */
  void Update(float z_meas_dist, float dt, int frame) {
    // 限制 dt 範圍，避免因為系統掉幀或時間戳錯誤導致 dt 過大或過小，讓矩陣數值爆炸
    dt = Clamp(dt, kDtMin, kDtMax);

    // 若尚未初始化，執行初始化流程
    if (!initialized_) {
      Initialize(z_meas_dist, dt, frame);
      return;
    }

    // 1. 構建狀態轉移矩陣 F(dt)
    // [1, dt]
    // [0,  1] -> 位置 = 位置 + 速度*dt
    kf_.transitionMatrix.at<float>(0, 1) = dt;

    // 2. 構建過程噪聲矩陣 Q(dt)
    // 採用 DWNA (Discrete White Noise Acceleration) 模型，符合物理運動規律
    UpdateProcessNoise(dt);

    // 3. 準備測量噪聲 R
    const float r = meas_std_m_ * meas_std_m_;

    // 4. 預測 (Predict)
    // 計算先驗狀態 (Prior State) 和 先驗協方差 (Prior Covariance)
    const cv::Mat pred = kf_.predict();
    const float d_pred = pred.at<float>(0); // 預測的距離

    // 5. 計算 Innovation (殘差) 與 NIS (Normalized Innovation Squared)
    // y = z - Hx (測量值 - 預測值)
    const float y   = z_meas_dist - d_pred;
    // S = HPH' + R (Innovation Covariance)
    const float S   = kf_.errorCovPre.at<float>(0, 0) + r; 
    // NIS = y' * S^-1 * y (在 1D 情況下就是 y^2 / S)
    // 這是一個卡方分佈變數，用來評估測量值與預測值的「一致性」
    const float nis = (S > kEps) ? (y * y / S) : 0.0f;

    bool updated = false;

    // 6. 根據 NIS 決定更新策略 (Gating Logic)
    if (nis <= gate_nis_soft_) {
      // [正常更新]：測量值很合理，完全信任 R
      updated = CorrectWithR(z_meas_dist, r);
    } else if (nis <= gate_nis_hard_) {
      // [軟更新 (Soft Update)]：測量值有點偏，但不至於太離譜。
      // 策略：放大 R (測量噪聲)，降低 Kalman Gain。
      // 效果：還是會更新狀態，但更新幅度變小，避免被雜訊拉偏，同時又能跟上真正的劇烈變動。
      updated = CorrectWithR(z_meas_dist, r * kSoftUpdateRScale);
    } else {
      // [硬拒絕 (Hard Reject)]：測量值偏差極大 (Outlier)，視為雜訊。
      // 策略：完全不使用此測量值，只使用預測值 (Predict) 作為當前狀態。
      Reject();
    }

    // 7. 更新後的狀態維護與復原機制 (Recovery Logic)
    if (updated) {
      // 如果成功更新，重置拒絕計數器
      rejection_count_ = 0;
      time_since_update_s_ = 0.0f;
      last_accepted_meas_dist_ = z_meas_dist;
      has_last_accepted_meas_ = true;
    } else {
      // 如果被拒絕，累加計數與時間
      rejection_count_++;
      time_since_update_s_ += dt;

      // [復原機制]：
      // 如果連續多次拒絕，或長時間沒更新，代表可能：
      // 1. 目標真的發生了劇烈位移（例如前車急煞後換車道）。
      // 2. 預測軌跡已經偏離太遠。
      // 必須強制「重置 (Reinitialize)」以跟上最新的測量值，避免永遠死鎖在舊軌跡。
      if (rejection_count_ >= kMaxConsecutiveRejections ||
          time_since_update_s_ >= kMaxNoUpdateSeconds) {
        ReinitializeFromMeasurement(z_meas_dist, dt, frame);
      }
    }

    last_frame_ = frame;
  }

  bool Initialized() const { return initialized_; }

  // 獲取平滑後的距離
  float Distance() const {
    return initialized_ ? kf_.statePost.at<float>(0) : 0.0f;
  }

  // 獲取估測的相對速度
  float RelSpeed() const {
    return initialized_ ? kf_.statePost.at<float>(1) : 0.0f;
  }

  int LastFrame() const { return last_frame_; }

private:
  // ---------- 常數定義 ----------
  static constexpr float kDtMin = 0.005f; // 最小 dt，避免除以 0
  static constexpr float kDtMax = 0.2f;   // 最大 dt，避免預測發散
  static constexpr float kEps   = 1e-6f;  // 極小值

  // 軟更新時 R 的放大倍率。放大 20 倍代表對測量值的信任度大幅降低。
  static constexpr float kSoftUpdateRScale = 20.0f;

  // 復原觸發條件
  static constexpr int   kMaxConsecutiveRejections = 3;     // 連續拒絕 3 次 -> 強制復原
  static constexpr float kMaxNoUpdateSeconds       = 0.3f;  // 或超過 0.3 秒沒更新 -> 強制復原

  // 重置時速度估算的物理極限保護（216 km/h），避免雜訊算出飛機般的速度
  static constexpr float kMaxAbsRelSpeedMps = 60.0f; 

  // ---------- 狀態變數 ----------
  cv::KalmanFilter kf_;

  bool  initialized_ = false;
  int   last_frame_  = -1;

  int   rejection_count_ = 0;           // 連續拒絕次數
  float time_since_update_s_ = 0.0f;    // 距上次有效更新的時間

  bool  has_last_accepted_meas_ = false;  // 是否有上一次可信的測量值（用於算速度）
  float last_accepted_meas_dist_ = 0.0f;

  // 可調參數 (Tuning Parameters)
  float meas_std_m_     = 0.6f;
  float accel_std_mps2_ = 2.0f;
  float gate_nis_soft_  = 9.0f;
  float gate_nis_hard_  = 25.0f;

private:
  // 輔助函式：數值夾止
  static float Clamp(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
  }

  // 初始化函式
  void Initialize(float z_meas_dist, float dt, int frame) {
    // 狀態維度: 2 (距離, 速度), 測量維度: 1 (距離), 控制維度: 0
    kf_.init(2, 1, 0, CV_32F);

    // 狀態轉移矩陣 F
    kf_.transitionMatrix    = (cv::Mat_<float>(2, 2) << 1, dt, 0, 1);
    // 測量矩陣 H
    kf_.measurementMatrix   = (cv::Mat_<float>(1, 2) << 1, 0);

    // 初始狀態：距離設為測量值，速度設為 0
    kf_.statePost           = (cv::Mat_<float>(2, 1) << z_meas_dist, 0.0f);
    
    // 初始 P 矩陣（誤差協方差）：給予較大的不確定性，讓 Filter 快速收斂
    kf_.errorCovPost        = cv::Mat::eye(2, 2, CV_32F) * 10.0f;

    kf_.processNoiseCov     = cv::Mat::eye(2, 2, CV_32F);
    kf_.measurementNoiseCov = (cv::Mat_<float>(1, 1) << (meas_std_m_ * meas_std_m_));

    initialized_ = true;
    last_frame_ = frame;

    rejection_count_ = 0;
    time_since_update_s_ = 0.0f;

    has_last_accepted_meas_ = true;
    last_accepted_meas_dist_ = z_meas_dist;
  }

  // 根據 dt 更新過程噪聲矩陣 Q
  // 這是 DWNA 模型的精髓，讓不確定性隨著時間差做物理上的縮放
  void UpdateProcessNoise(float dt) {
    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;
    const float sa2 = accel_std_mps2_ * accel_std_mps2_;

    kf_.processNoiseCov = (cv::Mat_<float>(2, 2) <<
      0.25f * dt4 * sa2, 0.5f * dt3 * sa2,
      0.5f  * dt3 * sa2,        dt2 * sa2
    );
  }

  // 執行修正 (Correct) 步驟
  bool CorrectWithR(float z_meas_dist, float r_used) {
    // 動態調整本次使用的測量噪聲 R
    kf_.measurementNoiseCov.at<float>(0, 0) = r_used;

    cv::Mat_<float> measurement(1, 1);
    measurement(0) = z_meas_dist;

    kf_.correct(measurement);
    return true;
  }

  // 拒絕更新
  void Reject() {
    // 保持預測狀態 (StatePre) 作為後驗狀態 (StatePost)
    // 也就是說：完全忽略這次的測量，認為上一刻的預測是對的
    kf_.statePost    = kf_.statePre.clone();
    kf_.errorCovPost = kf_.errorCovPre.clone();
  }

  // 強制重置 (Reinitialize)
  // 當 Filter 迷失或跟丟時使用
  void ReinitializeFromMeasurement(float z_meas_dist, float dt, int frame) {
    // 嘗試從上一次有效測量計算一個粗略的速度 (v = dx / dt)
    // 這比將速度直接歸零更好，能減少重置後的收斂時間
    float v0 = 0.0f;
    const float elapsed = std::max(dt, time_since_update_s_ + dt);

    if (has_last_accepted_meas_) {
      v0 = (z_meas_dist - last_accepted_meas_dist_) / elapsed;
      // 速度限幅保護
      v0 = Clamp(v0, -kMaxAbsRelSpeedMps, kMaxAbsRelSpeedMps);
    }

    // 重設狀態向量
    kf_.statePost.at<float>(0) = z_meas_dist;
    kf_.statePost.at<float>(1) = v0;

    // 重設 P 矩陣：給予極大的不確定性，讓 Filter 在接下來幾幀完全相信測量值
    kf_.errorCovPost = cv::Mat::eye(2, 2, CV_32F);
    kf_.errorCovPost.at<float>(0, 0) = 25.0f;  // 距離變異數
    kf_.errorCovPost.at<float>(1, 1) = 100.0f; // 速度變異數

    rejection_count_ = 0;
    time_since_update_s_ = 0.0f;

    has_last_accepted_meas_ = true;
    last_accepted_meas_dist_ = z_meas_dist;

    last_frame_ = frame;
  }
};

} // namespace acc