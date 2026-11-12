# Feasibility decision: **GO-FULL**

Rules applied exactly as pre-registered in PLAN.md. Decision uses M2-GP only.

## Data-quality gate

| Drive | Field | Heading RMS (deg) | GPS lag (s) | IMU lag (s) | Moving (s) | Result |
|---|---|---|---|---|---|---|
| 01_13B_Jackal | 13B | 0.85 | +0.07 | -0.00 | 414 | PASS |
| 02_13B_Jackal | 13B | 0.84 | +0.07 | -0.00 | 412 | PASS |
| 04_13D_Jackal | 13D | 0.92 | +0.07 | -0.00 | 410 | PASS |
| 05_13D_Jackal | 13D | 0.85 | +0.07 | -0.00 | 411 | PASS |


## Within-field results at H = 2 s (leave-one-drive-out; basis for G1, G2)

| Test drives | Windows | M0 (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |
|---|---|---|---|---|---|---|---|---|
| 4 | 3279 | 0.167 | 0.032 | 0.020 | 0.020 | +87.9% [+87.2%, +88.6%] | +36.3% [+32.8%, +39.4%] | +36.3% |

## Cross-field results at H = 2 s (train on one field, test on the other; basis for G4)

| Direction | Windows | M0 (m) | M0 turning (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |
|---|---|---|---|---|---|---|---|---|---|
| 13B->13D | 1636 | 0.206 | 0.263 | 0.082 | 0.083 | 0.084 | +59.6% [+58.9%, +60.3%] | -1.6% [-3.7%, +0.2%] | -3.0% |
| 13D->13B | 1643 | 0.130 | 0.193 | 0.094 | 0.085 | 0.084 | +34.3% [+32.2%, +36.3%] | +10.0% [+8.4%, +11.7%] | +10.6% |

## Rules

- **S0** (M0 error on turning windows ≥ 0.15 m, every cross-field direction): PASS
- **G1** (within-field GP vs M0 ≥ 30%, CI low ≥ 15%): PASS
- **G2** (within-field GP vs M1 ≥ 10%, CI low > 0): PASS
- **G4** (cross-field GP vs M1 ≥ 10%, CI low > 0, both directions; reported, not a gate): FAIL: offline model does not transfer across fields
- **G3** (adaptation signal): PASS
  - G3a mean cross-vs-within gap = +316.8% (needs ≥ 15%); per drive: 01_13B_Jackal +304.0%, 02_13B_Jackal +314.7%, 04_13D_Jackal +310.9%, 05_13D_Jackal +337.4%
  - G3b ICR parameter effect between fields (needs ≥ 0.05 m/s or rad/s at v=1.2, ω=0.8): a: effect 0.038 (CIs separated), b: effect 0.158 (CIs separated) OK, c: effect 0.030 (CIs separated)

## Calibrated ICR parameters (v_x = a·v_w, ω = b·ω_w, v_y = c·ω_w)

| Field | a | b | c |
|---|---|---|---|
| 13B | 0.955 [0.954, 0.956] | 0.534 [0.526, 0.544] | 0.082 [0.081, 0.083] |
| 13D | 0.923 [0.922, 0.924] | 0.337 [0.323, 0.352] | 0.120 [0.120, 0.121] |

## Next step

Proceed: learned-residual MPC with online updating vs offline-only; cross-field generalisation.
