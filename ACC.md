 先調 acc.max_accel_mps2。

  目前你的 YAML 是：

  acc:
    max_accel_mps2: 5.0

  這是 ACC 允許的最大加速度上限。若覺得加速不夠，可以先試：

  max_accel_mps2: 6.0
  # 或 7.0

  再看是否真的有變強。

  如果調了 max_accel_mps2 還是不夠，接著看這幾個：

  acc:
    jerk_limit_mps3: 2.0

  jerk_limit_mps3 限制加速度變化率。太低會讓加速「慢慢上來」，感覺油門不夠直接。可試 3.0 或 4.0。

  acc:
    cruise_speed_kmh: 30.0

  如果目前車速已接近 cruise_speed_kmh，ACC 會自然不再積極加速。要提高巡航目標速度就調這個。

  acc:
    use_external_ego_speed: 0

  如果你在實車測試，建議確認 ACC 是否用到正確 ego speed。速度估錯也會導致加速策略看起來很弱。

  建議調整順序：

  1. max_accel_mps2: 5.0 -> 6.0
  2. 若反應還是太鈍，jerk_limit_mps3: 2.0 -> 3.0
  3. 若只是目標速度太低，調高 cruise_speed_kmh

  另外實車還要確認 Speed/Brake 與 TX master 都有開，否則 ACC 算出速度命令也不一定會真的送到車上。
  
  再調
  stability.max_speed_rise_mps2、stability.max_jerk_acc_mps3、stability.speed_lowpass_alpha，或把油門 mapping 改成直接吃 acc_control_accel_cmd_mps2，否則單純加大
  acc.max_accel_mps2 效果會有限。
