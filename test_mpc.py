#!/usr/bin/env python3
"""
Step 3 checks for mpc.py.
    python test_mpc.py            # all five drives (about 10-15 min)
    python test_mpc.py --quick    # drive 04 only (writes summary_quick.csv; summary.csv stays the full run)
Writes results/mpc/summary.csv, results/mpc/sharpest_turn.png, results/mpc/speed_profile.png.

Plants and models follow the evaluation folds: the simulated robot of drive X uses X's own a, b;
the MPC uses a, b fitted on the other calibrated drives (fold "test_X"). Drives 01 and 03 have no
IMU data, so robot and MPC both use the all-drive calibration.
"""
import argparse
import os

import numpy as np
import pandas as pd

import mpc
import sim

OUT = "results/mpc"
TAU = 0.2
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def models(drive, cal):
    plant = cal["per_drive"].get(drive, cal["all"])
    model = cal["per_fold"].get(f"test_{drive}", cal["all"])
    return sim.Robot(plant["a"], plant["b"], TAU), mpc.MPC(model["a"], model["b"], TAU)


def sharpest_turn_start(P):
    """Progress 8 m before the point of highest |curvature| (a fair, hard test spot)."""
    i = int(np.argmax(np.abs(P.kappa)))
    return max(P.s[i] - 8.0, 0.0), P.s[i]


def start_state_at(P, s0, v):
    X = P.offset_pose(s0, 0.0)
    X[3] = v
    return X


def test_straight(cal):
    P = sim.Path("04_13D_Jackal")
    robot, ctrl = models(P.drive, cal)
    i = np.flatnonzero(P.is_row)[200]                     # 20 m into the first row
    log = sim.run(P, robot, ctrl, X0=start_state_at(P, P.s[i], 0.0), t_max=12.0, stop_at_end=False, s_init=P.s[i])
    late = log[log.t > 6]                                # after the speed-up
    check("1 straight row", abs(late.v_c.mean() - mpc.V_REF) < 0.05 and late.e_y.abs().max() < 0.05,
          f"after 6 s: commanded speed {late.v_c.mean():.2f} m/s (target {mpc.V_REF}), max |e_y| {late.e_y.abs().max():.3f} m")


def test_turn(cal):
    worst = []
    for d in sim.ALL_DRIVES:
        P = sim.Path(d)
        robot, ctrl = models(d, cal)
        s0, s_peak = sharpest_turn_start(P)
        log = sim.run(P, robot, ctrl, X0=start_state_at(P, s0, mpc.V_REF), t_max=25.0, stop_at_end=False, s_init=s0)
        near = log[(log.s > s_peak - 3) & (log.s < s_peak + 3)]
        inside = bool((log.e_y.abs() <= log.d).all())
        worst.append((d, float(np.abs(P.kappa).max()), float(log.e_y.abs().max()), float(near.v_c.min()), inside))
    for d, k, e, v, ok in worst:
        print(f"      {d}: sharpest |kappa| {k:.2f} 1/m -> max |e_y| {e:.3f} m, min speed near it {v:.2f} m/s, inside corridor {ok}")
    check("2 sharpest turn of every drive", all(w[4] for w in worst),
          f"stays inside the corridor in {sum(w[4] for w in worst)}/{len(worst)} drives")


def test_full(cal, drives, summary_name="summary.csv"):
    rows, logs = [], {}
    for d in drives:
        P = sim.Path(d)
        for name in ("mpc", "pure_pursuit"):
            robot, ctrl = models(d, cal)
            if name == "pure_pursuit":
                ctrl = sim.pure_pursuit
            log = sim.run(P, robot, ctrl)
            logs[(d, name)] = log
            r, t = log[log.is_row], log[~log.is_row]
            row = dict(drive=d, controller=name, reached_end=bool(log.s.iloc[-1] >= P.length - 0.5),
                       time_s=round(float(log.t.iloc[-1]), 1),
                       mean_true_speed=round(float(P.length / log.t.iloc[-1]), 3),
                       row_rms_ey=round(float(np.sqrt((r.e_y ** 2).mean())), 4), row_max_ey=round(float(r.e_y.abs().max()), 3),
                       turn_rms_ey=round(float(np.sqrt((t.e_y ** 2).mean())), 4), turn_max_ey=round(float(t.e_y.abs().max()), 3),
                       corridor_violation_pct=round(100 * float((log.e_y.abs() > log.d).mean()), 3))
            if name == "mpc":
                row.update(ctrl.summary())
            rows.append(row)
            print(f"      {d} {name:12s} done: {row['time_s']} s, row max |e_y| {row['row_max_ey']}, turn max |e_y| {row['turn_max_ey']}")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, summary_name), index=False)   # --quick never overwrites the full summary
    m = df[df.controller == "mpc"]
    check("3 MPC completes every path inside the corridor",
          bool(m.reached_end.all()) and float(m.corridor_violation_pct.max()) == 0.0,
          f"{int(m.reached_end.sum())}/{len(m)} completed, worst corridor violation {m.corridor_violation_pct.max()} % of steps")
    pp = df[df.controller == "pure_pursuit"].set_index("drive")
    mm = m.set_index("drive")
    faster = (mm.time_s < pp.time_s).all()
    print(df[["drive", "controller", "time_s", "mean_true_speed", "row_rms_ey", "row_max_ey", "turn_rms_ey", "turn_max_ey"]].to_string(index=False))
    check("4 MPC vs pure pursuit", bool(faster),
          f"MPC finishes faster on {int((mm.time_s < pp.time_s).sum())}/{len(mm)} drives "
          f"(total {mm.time_s.sum():.0f} s vs {pp.time_s.sum():.0f} s)")
    check("5 solve time", float(m.solve_ms_p99.max()) < 100.0 and int(m.failures.sum()) == 0,
          f"median {m.solve_ms_median.median():.1f} ms, worst p99 {m.solve_ms_p99.max():.1f} ms, "
          f"worst max {m.solve_ms_max.max():.0f} ms, failures {int(m.failures.sum())}, slack steps {int(m.slack_steps.sum())} "
          f"(control period 100 ms)")
    return logs


def figures(cal, logs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # sharpest turn over all drives, MPC vs pure pursuit
    best = max(sim.ALL_DRIVES, key=lambda d: np.abs(sim.Path(d).kappa).max())
    P = sim.Path(best)
    s0, s_peak = sharpest_turn_start(P)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    sl = (P.s > s0 - 2) & (P.s < s_peak + 14)
    ax[0].plot(P.x[sl], P.y[sl], color="0.75", lw=5, label="reference")
    for name, c in (("mpc", "tab:blue"), ("pure_pursuit", "tab:orange")):
        robot, ctrl = models(best, cal)
        log = sim.run(P, robot, ctrl if name == "mpc" else sim.pure_pursuit,
                      X0=start_state_at(P, s0, mpc.V_REF if name == "mpc" else 1.4), t_max=20, stop_at_end=False, s_init=s0)
        w = log[log.s < s_peak + 12]
        ax[0].plot(w.x, w.y, color=c, lw=1.3, label=name.replace("_", " "))
        ax[1].plot(w.s - s_peak, w.e_y, color=c, lw=1.3, label=f"{name.replace('_', ' ')}: e_y")
    ax[1].plot(P.s[sl] - s_peak, P.d[sl], "r--", lw=1, label="corridor ±d")
    ax[1].plot(P.s[sl] - s_peak, -P.d[sl], "r--", lw=1)
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"{best}: sharpest turn (|κ| = {np.abs(P.kappa).max():.2f} 1/m)")
    ax[1].set(xlabel="distance from the sharpest point (m)", ylabel="e_y (m)", ylim=(-2.3, 2.3))
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sharpest_turn.png"), dpi=110); plt.close(fig)
    # speed profile along one drive
    d = "04_13D_Jackal" if ("04_13D_Jackal", "mpc") in logs else next(iter(logs))[0]
    fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    for name, c in (("mpc", "tab:blue"), ("pure_pursuit", "tab:orange")):
        log = logs[(d, name)]
        ax[0].plot(log.s, log.v_c, color=c, lw=0.8, label=name.replace("_", " "))
        ax[1].plot(log.s, log.e_y, color=c, lw=0.8)
    P = sim.Path(d)
    for a in ax:
        a.fill_between(P.s, -5, 5, where=~P.is_row, color="tab:red", alpha=0.08, step="mid")
    ax[0].set(ylabel="v_c (m/s)", ylim=(0, 1.9), title=f"{d}: speed and lateral error along the path (turns shaded)")
    ax[1].set(ylabel="e_y (m)", xlabel="s (m)", ylim=(-0.2, 0.2))
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "speed_profile.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cal = sim.calibration_table()
    test_straight(cal)
    test_turn(cal)
    logs = test_full(cal, ["04_13D_Jackal"] if args.quick else list(sim.ALL_DRIVES),
                     summary_name="summary_quick.csv" if args.quick else "summary.csv")
    figures(cal, logs)
    print(f"\n{sum(results)}/{len(results)} tests passed")
