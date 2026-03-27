 python3 tools/update_sensing_yaml_from_ground_csv.py \
    --intrinsics-yaml ./tools/Geometry/calib_out/camera_intrinsics.yaml \
    --ground-csv Camera-Config/單應性1280x720.csv \
    --output-yaml Camera-Config/Sensing-3M.yaml
    
    如果你下次的 CSV 橫向定義改成「左為正」，再加 --invert-lateral-sign
