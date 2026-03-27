python calibrate_intrinsics_charuco.py \
  --input=./Picture/ \
  --dict DICT_4X4_50 \
  --squares_x 7 --squares_y 5 \
  --square_len 0.035 --marker_len 0.035 \
  --reject_outliers --save_used
  
----------------------------------

python calibrate_intrinsics.py \
  --input=./Picture/ \
  --cols 8 --rows 6 --square 0.035 \
  --min_sharpness 0 --min_coverage 0 \
  --reject_outliers --save_used

