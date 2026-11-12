# Slip-residual feasibility check (CitrusFarm, Jackal)

Go/no-go test for the term project. Read `PLAN.md` first: it fixes the rules **before** the real
data is run. The script applies those rules and writes `results/decision.md`.

## 1. Install

```bash
pip install rosbags numpy scipy pandas scikit-learn matplotlib
```

You do not need a ROS installation; `rosbags` reads ROS1 bags directly.

## 2. Download (only what is needed)

Ground truth exists for drives 01–05. Fields: 13B = 01, 02, 03; 13D = 04, 05.

**Minimum:** drives 01, 04, 05. This allows a within-field test on 13D only.
**Recommended:** all five drives. Then both fields get a within-field test, which makes G1, G2 and G3 much more trustworthy.

```bash
B=s3://ucr-robotics/citrus-farm-dataset
D=/path/to/citrus-farm-dataset          # same as data_root in config.json
for s in 01_13B_Jackal 02_13B_Jackal 03_13B_Jackal 04_13D_Jackal 05_13D_Jackal; do
  aws s3 sync --no-sign-request $B/ground_truth/$s $D/ground_truth/$s        # gt.bag, gt.csv (small)
  aws s3 sync --no-sign-request $B/$s $D/$s --exclude "*" --include "odom_*.bag"   # ~47 MB each
  aws s3 sync --no-sign-request $B/$s $D/$s --exclude "*" --include "base_*.bag"   # IMU (large: also holds LiDAR)
done
# calibration (tiny) for the GPS lever arm:
aws s3 sync --no-sign-request $B/Calibration/results $D/Calibration/results
```

**IMU source.** The script looks for the MicroStrain IMU in `base_*.bag` first. If those files are not present, it uses the ZED IMU from any `zed_*.bag` you already have. You therefore need either the base bags or the zed bags of a drive, not both. Use the same source for all drives if you can.

## 3. Set the GPS lever arm (required)

`lever_arm_xy` is the GPS antenna position (x forward, y left, in metres) in `base_link`. Compose it from:
- `Calibration/results/05-baselink-lidar-result.txt`
- `Calibration/results/04-lidar-gps-result.txt`

That is, T_base_gps = T_base_lidar · T_lidar_gps. Take only the translation x and y.

This cannot be estimated from driving data. An antenna x-offset and skid-steer lateral slip produce the same signal, so a wrong lever arm silently corrupts the lateral-slip result.

If you want to see the pipeline run before you have the lever arm, set `"allow_zero_lever_arm": true`. The result is then stamped **PROVISIONAL**.

## 4. Run

```bash
python feasibility.py --config config.json --quick   # smoke test, ~2 min, NOT for the decision
python feasibility.py --config config.json           # the real run, roughly 10-30 min on a laptop
```

Outputs go to `results/`:
- `decision.md`: the verdict, every rule with its numbers, and the next step
- `data_quality.csv`
- `errors_by_model.csv`: per drive, model, horizon, and straight vs turning windows
- `icr_params.csv`
- `summary.json`
- `figures/`: `slip_vs_omega.png`, `slip_per_drive.png`, `error_vs_horizon.png`, `example_rollout.png`

## 5. Self-test (already done; repeat if you change the code)

`selftest_synthetic.py` writes fake drives in the exact CitrusFarm bag format with a **known** slip model. It deliberately includes:
- an upside-down, rotated IMU with gyro bias,
- a GPS lever arm,
- a 0.08 s GPS timestamp lag,
- 1 cm RTK noise.

```bash
python selftest_synthetic.py --out synth_strong --slip strong && python feasibility.py --config synth_strong/config.json
python selftest_synthetic.py --out synth_noslip --slip no     && python feasibility.py --config synth_noslip/config.json
```

The strong-slip case should give a GO verdict. The no-slip case must give NO-GO.

## Notes and known risks

- **Jackal odometry may already compensate part of the slip.** Clearpath's stock `jackal_control/config/control.yaml` sets `wheel_separation_multiplier: 1.5`. It is not documented whether CitrusFarm kept that default. In that case ω_w is already roughly scaled, the M1 parameter b comes out near 1, and the remaining slip is only the nonlinear, terrain-dependent part. S0 tests whether that remainder is big enough to matter.
- **Wheel odometry stands in for the command.** No `cmd_vel` was recorded, so the learned model maps "what the wheels did" to "what the robot did". That is the right mapping for MPC, provided the MPC commands wheel speeds that the Jackal's velocity loop tracks well.
- If a drive fails the quality gate, look at its row in `data_quality.csv` before changing anything. Any rule change goes into the PLAN.md amendments section with date and reason.
