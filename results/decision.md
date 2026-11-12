# Feasibility decision: **NO-GO**

Rules applied exactly as pre-registered in PLAN.md. Decision uses M2-GP only.

## Data-quality gate

| Drive | Field | Heading RMS (deg) | GPS lag (s) | IMU lag (s) | Moving (s) | Result |
|---|---|---|---|---|---|---|
| 01_13B_Jackal | 13B | nan | +nan | +nan | nan | FAIL |
| 02_13B_Jackal | 13B | 1.26 | -0.03 | -0.10 | 435 | PASS |
| 03_13B_Jackal | 13B | nan | +nan | +nan | nan | FAIL |
| 04_13D_Jackal | 13D | 1.31 | -0.05 | -0.10 | 506 | PASS |
| 05_13D_Jackal | 13D | 1.24 | -0.03 | -0.12 | 503 | PASS |

Flags:
- 01_13B_Jackal: error: no IMU found in /home/symum2026/citrusfarm/01_13B_Jackal (looked for ['microstrain', 'zed'])
- 03_13B_Jackal: error: no IMU found in /home/symum2026/citrusfarm/03_13B_Jackal (looked for ['microstrain', 'zed'])


## Within-field results at H = 2 s (leave-one-drive-out; basis for G1, G2)

| Test drives | Windows | M0 (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |
|---|---|---|---|---|---|---|---|---|
| 2 | 2011 | 0.291 | 0.052 | 0.054 | 0.056 | +81.5% [+80.2%, +82.9%] | -3.5% [-6.9%, -0.1%] | -8.2% |

## Cross-field results at H = 2 s (train on one field, test on the other; basis for G4)

| Direction | Windows | M0 (m) | M0 turning (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |
|---|---|---|---|---|---|---|---|---|---|
| 13B->13D | 2011 | 0.291 | 0.270 | 0.052 | 0.056 | 0.058 | +80.6% [+79.3%, +82.0%] | -8.3% [-13.4%, -3.0%] | -11.4% |
| 13D->13B | 871 | 0.301 | 0.282 | 0.051 | 0.052 | 0.050 | +82.6% [+79.8%, +84.8%] | -1.8% [-8.0%, +3.5%] | +3.0% |

## Rules

- **S0** (M0 error on turning windows ≥ 0.15 m, every cross-field direction): PASS
- **G1** (within-field GP vs M0 ≥ 30%, CI low ≥ 15%): PASS
- **G2** (within-field GP vs M1 ≥ 10%, CI low > 0): FAIL
- **G4** (cross-field GP vs M1 ≥ 10%, CI low > 0, both directions; reported, not a gate): FAIL: offline model does not transfer across fields
- **G3** (adaptation signal): FAIL
  - G3a mean cross-vs-within gap = +4.7% (needs ≥ 15%); per drive: 04_13D_Jackal +6.4%, 05_13D_Jackal +3.1%
  - G3b ICR parameter effect between fields (needs ≥ 0.05 m/s or rad/s at v=1.2, ω=0.8): a: effect 0.001, b: effect 0.016, c: effect 0.001

## Calibrated ICR parameters (v_x = a·v_w, ω = b·ω_w, v_y = c·ω_w)

| Field | a | b | c |
|---|---|---|---|
| 13B | 0.885 [0.884, 0.887] | 0.989 [0.977, 1.001] | -0.014 [-0.026, 0.001] |
| 13D | 0.886 [0.885, 0.887] | 0.969 [0.959, 0.979] | -0.016 [-0.024, -0.009] |

## Next step

Switch to idea #2: neural approximation of MPC with a safety projection.
