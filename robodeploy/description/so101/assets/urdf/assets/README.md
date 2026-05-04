# SO-101 mesh assets (STL)

Place the STL files exported with the canonical URDF (`../so101.urdf`) in this directory.
Paths in the URDF are `assets/<filename>.stl` relative to the URDF file.

Expected files (13 unique):

- `base_motor_holder_so101_v1.stl`
- `base_so101_v2.stl`
- `sts3215_03a_v1.stl`
- `waveshare_mounting_plate_so101_v2.stl`
- `motor_holder_so101_base_v1.stl`
- `rotation_pitch_so101_v1.stl`
- `upper_arm_so101_v1.stl`
- `under_arm_so101_v1.stl`
- `motor_holder_so101_wrist_v1.stl`
- `sts3215_03a_no_horn_v1.stl`
- `wrist_roll_pitch_so101_v2.stl`
- `wrist_roll_follower_so101_v1.stl`
- `moving_jaw_so101_v1.stl`

If any file is missing, `SO101Description.asset_path(URDF)` returns a generated URDF with small box placeholders instead of meshes so MuJoCo and RViz still load.
