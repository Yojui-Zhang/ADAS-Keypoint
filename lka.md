• 先調這幾個 LKA 參數，優先順序如下：

  1. lka.softening
     目前 Stanley 橫向修正是：

  atan2(k * cte, velocity_mps + softening)

  速度越高，分母越大，控制力道自然變小。
  若高速時覺得太弱，可以先把：

  softening: 0.5

  略降，例如 0.3 或 0.2。
  但這會同時讓低速也更敏感，可能增加低速左右擺動。

  2. lka.k_straight
     如果主要是直線或微彎高速跟線偏弱，調這個：

  k_straight: 0.7

  可以試：

  k_straight: 0.9
  # 或 1.0

  這是最直接增加 Stanley feedback 的方式。

  3. lka.k_curve
     如果是彎道高速跟不住，調這個：

  k_curve: 3.0

  可往上試，例如 3.5、4.0。
  但彎道太高容易過衝或抖動。

  4. lka.max_steer_rate_deg_s
     如果你感覺不是「角度不夠」，而是「方向盤跟得太慢」，調這個：

  max_steer_rate_deg_s: 200.0

  可以試 250 或 300。
  它限制每秒方向角變化速度。

  5. stability.steer_high_speed_guard_kmh
     如果高於某速度後明顯被削弱，也檢查 Stability 層：

  steer_high_speed_guard_kmh: 60.0

  若你在接近或超過 60 km/h 測試，Supervisor 可能會保守限制方向輸出。

  建議先只改一個：
  高速直線偏弱：先把 k_straight: 0.7 -> 0.9。
  高速彎道偏弱：先把 k_curve: 3.0 -> 3.5。
  反應太慢：再加 max_steer_rate_deg_s。
