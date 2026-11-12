# Feasibility decision: **NO-GO**

Rules applied exactly as pre-registered in PLAN.md. Decision uses M2-GP only.

> **PROVISIONAL RESULT, NOT VALID FOR THE DECISION:**
> - --quick run: reduced GP size and bootstrap; not valid for the decision

## Data-quality gate

| Drive | Field | Heading RMS (deg) | GPS lag (s) | IMU lag (s) | Moving (s) | Result |
|---|---|---|---|---|---|---|
| 01_13B_Jackal | 13B | 0.81 | +0.08 | -0.00 | 414 | PASS |
| 02_13B_Jackal | 13B | 0.83 | +0.07 | -0.00 | 412 | PASS |
| 04_13D_Jackal | 13D | 0.86 | +0.07 | -0.00 | 410 | PASS |
| 05_13D_Jackal | 13D | 0.79 | +0.07 | -0.00 | 411 | PASS |


## Within-field results at H = 2 s (leave-one-drive-out; basis for G1, G2)

| Test drives | Windows | M0 (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |
|---|---|---|---|---|---|---|---|---|
| 4 | 3279 | 0.019 | 0.019 | 0.021 | 0.020 | -7.8% [-9.4%, -6.2%] | -12.1% [-14.0%, -10.5%] | -9.2% |

## Cross-field results at H = 2 s (train on one field, test on the other; basis for G4)

| Direction | Windows | M0 (m) | M0 turning (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |
|---|---|---|---|---|---|---|---|---|---|
| 13B->13D | 1636 | 0.019 | 0.021 | 0.018 | 0.020 | 0.019 | -6.7% [-9.9%, -4.1%] | -10.8% [-14.0%, -7.5%] | -6.1% |
| 13D->13B | 1643 | 0.020 | 0.025 | 0.019 | 0.021 | 0.020 | -4.9% [-6.7%, -3.3%] | -9.6% [-11.6%, -7.5%] | -5.6% |

## Rules

- **S0** (M0 error on turning windows ≥ 0.15 m, every cross-field direction): FAIL
- **G1** (within-field GP vs M0 ≥ 30%, CI low ≥ 15%): FAIL
- **G2** (within-field GP vs M1 ≥ 10%, CI low > 0): FAIL
- **G4** (cross-field GP vs M1 ≥ 10%, CI low > 0, both directions; reported, not a gate): FAIL: offline model does not transfer across fields
- **G3** (adaptation signal): FAIL
  - G3a mean cross-vs-within gap = -1.5% (needs ≥ 15%); per drive: 01_13B_Jackal -9.4%, 02_13B_Jackal +1.6%, 04_13D_Jackal +0.1%, 05_13D_Jackal +1.8%
  - G3b ICR parameter effect between fields (needs ≥ 0.05 m/s or rad/s at v=1.2, ω=0.8): a: effect 0.000, b: effect 0.000 (CIs separated), c: effect 0.002 (CIs separated)

## Calibrated ICR parameters (v_x = a·v_w, ω = b·ω_w, v_y = c·ω_w)

| Field | a | b | c |
|---|---|---|---|
| 13B | 0.999 [0.998, 0.999] | 1.000 [0.999, 1.000] | 0.012 [0.010, 0.013] |
| 13D | 0.999 [0.998, 0.999] | 1.000 [1.000, 1.000] | 0.008 [0.007, 0.009] |

## Next step

Switch to idea #2: neural approximation of MPC with a safety projection.
