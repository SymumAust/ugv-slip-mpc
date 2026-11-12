# Pre-registered feasibility plan: learned slip residual for a skid-steer orchard UGV

**Project:** MO89905 term project (Prof. Kyunghwan Choi), learning-based MPC for an agricultural UGV
**Plan frozen:** 2026-11-13, before any real CitrusFarm data is run
**Decision deadline:** 2026-11-18 (proposal due 2026-11-22)

This file is written before looking at the real results. Thresholds, models, splits and the
primary metric are fixed here. `feasibility.py` applies these rules automatically and writes
`results/decision.md`. If any rule is changed after seeing data, the change must be logged in
the "Amendments" section at the bottom, with date and reason. It must not be edited silently.

## 1. Question

Does a learned slip residual predict the robot's short-horizon motion (1–2 s, the MPC
horizon) clearly better than (a) the nominal kinematic model and (b) a calibrated physics
(ICR) slip model? Does it do this on a field it was **not trained on**?

## 2. Data

CitrusFarm dataset (Teng et al., 2023), Clearpath Jackal (skid-steer). Ground truth exists for
sequences 01–05 only:

| Drive | Field | Use |
|---|---|---|
| 01_13B_Jackal | 13B | required |
| 02_13B_Jackal | 13B | optional (strengthens the within-field test) |
| 03_13B_Jackal | 13B | optional |
| 04_13D_Jackal | 13D | required |
| 05_13D_Jackal | 13D | required |

Sequences 06/07 (field 14B) have no ground truth and are not used. This gives **two fields**,
not three, so the cross-field test is 13B ↔ 13D.

Signals used:
- **Wheel odometry:** `/jackal_velocity_controller/odom`, 50 Hz. This gives v_w, ω_w and stands in for the command; the robot was teleoperated and no `cmd_vel` was recorded.
- **IMU gyro:** `/microstrain/imu/data` from `base_*.bag`, or `/zed2i/zed_node/imu/data` from `zed_*.bag` as fallback. This gives the true yaw rate.
- **RTK-GPS ground truth:** `/gps/fix/odometry` from `gt.bag`, or `gt.csv`. This gives the antenna position at 10 Hz (fixed RTK).
- **GPS antenna lever arm** (l_x, l_y) in `base_link`, from the dataset calibration files (CAD). This is required. Lateral skid-steer slip (ICR offset) and an x-offset of the antenna produce the *same* signature (lateral velocity ∝ ω), so the lever arm cannot be estimated from motion data. It must be taken from calibration.

## 3. Preprocessing (fixed)

1. **IMU up-axis:** the unit vector of the mean accelerometer reading. Yaw rate = gyro · up. This needs no assumption about how the IMU is mounted.
2. **Gyro bias:** the median gyro yaw rate over stationary periods (0.5 s moving median of |v_w| < 0.01 m/s and of |ω_w| < 0.01 rad/s, for ≥ 1 s). If there are none, bias = 0, and this is flagged.
3. **Time alignment:** the GPS and IMU lags relative to odometry are estimated by cross-correlation (GPS speed vs |v_w|; IMU yaw rate vs ω_w), searching ±5 s. They are applied if |lag| > 0.02 s.
4. **Common timeline:** 50 Hz over the overlap of all three streams. Any GPS gap > 0.5 s marks the surrounding samples as invalid.
5. **GPS velocity:** Savitzky–Golay derivative at 10 Hz (window 7 = 0.7 s, order 2), interpolated to 50 Hz.
6. **Heading ψ:** integrated bias-corrected yaw rate plus a slowly varying offset. The offset is fitted to the GPS course over ground on straight, moving segments (|v_w| > 0.5 m/s, |ω_w| < 0.05 rad/s), using a median in 60 s bins and linear interpolation between bins. Reverse driving is handled by adding π.
7. **True body velocity at `base_link`:** the antenna velocity is rotated into the body frame and the lever-arm term is removed: v_x = v_ax + ω l_y, v_y = v_ay − ω l_x.
8. **Slip targets:** Δv_x = v_x − v_w, v_y, Δω = ω_true − ω_w.

## 4. Models (fixed)

All models map wheel-odometry features to body velocities [v_x, v_y, ω]. None of them sees the IMU or GPS at prediction time.

| ID | Model | Parameters |
|---|---|---|
| **M0** | Nominal unicycle: [v_w, 0, ω_w] | none |
| **M1** | Calibrated physics (ICR/linear slip): v_x = a·v_w, ω = b·ω_w, v_y = c·ω_w | 3, least squares |
| **M2-GP** (primary learned) | M1 + Gaussian-process residual (ARD-RBF + white noise, one GP per output, 2000 training points) | — |
| M2-MLP (secondary) | M1 + MLP residual (2×64, early stopping, mean of 3 seeds) | — |

Learned-model features: [v_w, ω_w, dv_w/dt, dω_w/dt].
Training samples: valid, moving samples (|v_w| ≥ 0.05 m/s or |ω_w| ≥ 0.05 rad/s), downsampled to 10 Hz.

**Only M2-GP is used for the decision.** M2-MLP is reported but cannot rescue a failed rule. This avoids choosing the best model after seeing the results.

## 5. Primary metric

**Multi-step open-loop prediction error.**
- Rollout start points are taken every 0.5 s on the test drive, at valid moving samples. Each rollout starts from the true state: the GPS position, moved to `base_link` using ψ and the lever arm, and the fused heading ψ.
- Each model integrates its predicted body velocities from the measured wheel odometry, using Euler steps of 0.02 s.
- Error is measured at the GPS antenna point at horizons H = 0.5, 1, 2 s.

**Primary metric:** mean position error at **H = 2 s**.
Secondary metrics: H = 1 s, the mean heading error, and a breakdown into straight vs turning windows. A window counts as turning if max |ω_w| in it is > 0.3 rad/s.

**Improvement of model A over model B:** 1 − e_A / e_B.
Uncertainty is a block bootstrap (10 s blocks, 1000 resamples, seed 0) giving a 95 % CI.

## 6. Splits (fixed)

- **Within-field (primary for G1, G2):** leave one drive out inside a field. For each drive d, train on the other available drives of the *same* field and test on d. All within-field test windows are pooled. This needs ≥ 2 drives in at least one field.
- **Cross-field (G3, G4, S0):** train on all available 13B drives, test on each 13D drive. Then train on all 13D drives, test on each 13B drive. Errors are pooled over windows within each direction.

## 7. Decision rules (fixed; amended 2026-11-13, see Amendments)

**Data-quality gate (per drive).** A drive that fails is dropped and reported:
- **Q1:** heading-alignment residual RMS < 3°
- **Q2:** |GPS lag| and |IMU lag| < 0.5 s
- **Q3:** ≥ 5 min of valid moving data

**S0 (slip matters):** M0 mean position error at H = 2 s, on **turning windows**, is ≥ 0.15 m in every cross-field direction.

**G1 (learned beats nominal):** within-field M2-GP improvement over M0 at H = 2 s is ≥ 30 %, with bootstrap CI lower bound ≥ 15 %.

**G2 (learned beats physics):** within-field M2-GP improvement over M1 at H = 2 s is ≥ 10 %, with bootstrap CI lower bound > 0.

**G3 (adaptation has something to adapt to):** on the same test drive, the cross-field M2-GP is ≥ 15 % worse than the within-field M2-GP (mean over drives where both exist). Alternatively, at least one M1 parameter differs between the two fields with a **physical effect** ≥ 0.05 m/s (for a, at v = 1.2 m/s; for c, at ω = 0.8 rad/s) or ≥ 0.05 rad/s (for b, at ω = 0.8 rad/s), **and** the two fields' 95 % bootstrap CIs for that parameter do not overlap. Over a 2 s horizon, 0.05 m/s ≈ 10 cm and 0.05 rad/s ≈ 6°. Statistical separation alone is not enough: with ~1 h of data, even a practically irrelevant difference would be "significant".

**G4 (offline model transfers; reported, not a gate):** cross-field M2-GP improvement over M1 at H = 2 s is ≥ 10 %, with CI lower bound > 0, in both directions. If G4 fails while G1–G3 pass, online updating is *necessary*, not just a bonus. That is the strongest version of the paper's claim.

**Outcomes:**

| Outcome | Condition | Project direction |
|---|---|---|
| **GO-FULL** | Q, S0, G1, G2, G3 | Learned-residual MPC with online updating vs offline-only, cross-field generalisation |
| **GO-OFFLINE** | Q, S0, G1, G2, not G3 | Learned-residual MPC; online updating only as an ablation, not a claim |
| **PIVOT-ADAPTIVE** | Q, S0, not G2, G3 | Online adaptive ICR-parameter MPC (physics model, parameters learned online) |
| **NO-GO** | not S0; or not G2 and not G3; or G2 but not G1; or no usable within-field pair | Switch to idea #2: neural approximation of MPC with a safety projection |

## 8. Outputs

`results/`:
- `decision.md` (verdict with every rule and its numbers)
- `summary.json`
- `errors_by_model.csv`
- `data_quality.csv`
- `icr_params.csv`
- `figures/` (slip vs ω_w, per-drive slip, error vs horizon, example U-turn rollout)

## Amendments

**2026-11-13, before any real data was run.** Found by running the pipeline on synthetic bags with a known, strongly nonlinear, field-dependent slip model (`selftest_synthetic.py`).

1. **G1/G2 moved from cross-field to within-field; cross-field kept as G4 (reported).** On synthetic data with clearly learnable slip, the 13B-trained residual was *worse* than physics on 13D, because it learned 13B's curve exactly. The cross-field test was mixing two questions: "does learning beat physics?" and "does slip shift between fields?". G2 now answers the first with like-for-like training data. G3/G4 answer the second.
2. **S0 judged on turning windows.** With 32 % yaw-rate slip in the synthetic data, the all-window M0 error was only 0.10–0.15 m, because straight rows dominate the windows. Slip lives in turns, so S0 now looks there.
3. **Stationary detection uses a 0.5 s moving median.** Raw thresholding missed a 5 s standstill because of odometry noise.
4. **G3b requires a physically meaningful parameter difference**, not only separated CIs. This was first written as "≥ 10 % relative difference". The no-slip control run showed that breaks near zero: c = 0.012 vs 0.008 counted as a 27 % difference. It is now an effect threshold in m/s or rad/s at typical driving.

None of these thresholds was tuned to make the synthetic case pass. The synthetic run is only a pipeline check.
