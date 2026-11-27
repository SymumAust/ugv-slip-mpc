#!/usr/bin/env python3
"""
Step 6 checks for safety_filter.py.
    python test_filter.py                 # all four tests (test 4 drives every path in all three folds)
    python test_filter.py --smoke --out /tmp/f   # tests 1-3 only (a few seconds), for checking the code
Writes <out>/summary.csv, <out>/bad_command.png, <out>/edge_start.png  (default out = results/filter)

  1 safe command        a safe command passes unchanged (filter inactive)
  2 deliberately bad    "always turn left at full rate": without the filter the robot leaves the corridor,
                        with it the robot stays inside
  3 edge start          start 0.8 m off the path (limit 1.0) heading outwards, and 1.3 m off (outside);
                        the filter keeps / brings the robot inside
  4 full paths          DAgger network (round 5) with and without the filter on every fold's test drive
                        and the extra paths: tracking, corridor, how often and how much the filter acts, time
"""
import argparse
import os

import numpy as np
import pandas as pd

import imitation as im
import policy
import safety_filter as sfm
import sim

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def fold_parts(test, cal):
    m = cal["per_fold"][im.fold_name(test)]
    return sfm.SafetyFilter(m["a"], m["b"], im.TAU), im.plant(test, cal)


def row_start(P, offset=0.0, heading=0.0, v=1.0):
    i = np.flatnonzero(P.is_row)[150]                          # 15 m into the first row
    X = P.offset_pose(P.s[i], offset)
    X[2] += heading
    X[3] = v
    return X, P.s[i]


def const(u):
    return lambda P, X, info: np.array(u, float)


def test_safe(cal):
    P = sim.Path("04_13D_Jackal")
    filt, robot = fold_parts(P.drive, cal)
    X0, s0 = row_start(P)
    ctrl = sfm.SafeController(const([1.0, 0.0]), filt)
    ctrl.u_prev = np.array([1.0, 0.0])                         # already driving at 1.0 m/s
    sim.run(P, robot, ctrl, X0=X0, s_init=s0, t_max=5.0, stop_at_end=False)
    s = filt.summary()
    check("1 safe command passes unchanged", s["filter_active_pct"] == 0.0,
          f"filter active in {s['filter_active_pct']} % of {s['filter_steps']} steps")


def test_bad(cal, out):
    P = sim.Path("04_13D_Jackal")
    filt, robot = fold_parts(P.drive, cal)
    X0, s0 = row_start(P, v=1.6)
    bad = const([1.6, 1.0])                                    # full speed, full left turn, forever
    raw = sim.run(P, robot, bad, X0=X0, s_init=s0, t_max=6.0, stop_at_end=False)
    sc = sfm.SafeController(bad, filt); sc.u_prev = np.array([1.6, 0.0])
    safe = sim.run(P, robot, sc, X0=X0, s_init=s0, t_max=6.0, stop_at_end=False)
    s = filt.summary()
    ok = raw.e_y.abs().max() > 1.0 and safe.e_y.abs().max() <= 1.0 + 0.02
    check("2 deliberately bad command", bool(ok),
          f"without filter max |e_y| {raw.e_y.abs().max():.2f} m (corridor 1.0 m); with filter {safe.e_y.abs().max():.3f} m, "
          f"filter active {s['filter_active_pct']} %, slack steps {s['slack_steps']}, QP failures {s['qp_failures']}")
    plot_pair(raw, safe, P, os.path.join(out, "bad_command.png"),
              "Deliberately bad command (full speed, full left turn): without vs with the safety filter")


def test_edge(cal, out):
    pol = policy.Policy.load(os.path.join(im.MODELS, "dagger_test_04_13D_Jackal_r5.json"))
    P = sim.Path("04_13D_Jackal")
    logs, msgs, oks = [], [], []
    # 0.8 m / 0.15 rad is recoverable; 0.9 m / 0.25 rad is not (even the best control drifts ~0.2 m
    # further out under the turn-rate ramp and actuator lag) -> that one is a step-7 failure case
    for off, head in ((0.8, 0.15), (1.3, 0.15)):
        filt, robot = fold_parts(P.drive, cal)
        X0, s0 = row_start(P, offset=off, heading=head, v=1.4)
        pol.reset(); pol.u_prev = np.array([1.4, 0.0])
        raw = sim.run(P, robot, pol, X0=X0, s_init=s0, t_max=8.0, stop_at_end=False)
        pol.reset(); pol.u_prev = np.array([1.4, 0.0])
        ctrl = sfm.SafeController(pol, filt); ctrl.u_prev = np.array([1.4, 0.0])
        safe = sim.run(P, robot, ctrl, X0=X0, s_init=s0, t_max=8.0, stop_at_end=False)
        s = filt.summary()
        if off < 1.0:
            ok = safe.e_y.abs().max() <= 1.0 + 0.02
            msgs.append(f"start {off} m: max |e_y| network alone {raw.e_y.abs().max():.3f} m, with filter {safe.e_y.abs().max():.3f} m")
        else:
            back = safe.t[safe.e_y.abs() <= 1.0]
            ok = len(back) > 0 and safe[safe.t > back.iloc[0]].e_y.abs().max() <= 1.0 + 0.02
            msgs.append(f"start {off} m (outside): back inside after {back.iloc[0] if len(back) else np.nan:.1f} s "
                        f"(network alone {raw.t[raw.e_y.abs() <= 1.0].iloc[0] if (raw.e_y.abs() <= 1.0).any() else np.nan:.1f} s), "
                        f"recovery steps {s['recovery_steps']}")
        oks.append(bool(ok))
        logs.append((off, raw, safe))
    check("3 edge starts", all(oks), "; ".join(msgs))
    plot_edge(logs, P, os.path.join(out, "edge_start.png"))


def test_full(cal, out):
    rows = []
    for test in sim.CALIB_DRIVES:
        fold = im.fold_name(test)
        pol = policy.Policy.load(os.path.join(im.MODELS, f"dagger_{fold}_r5.json"))
        m = cal["per_fold"][fold]
        for drive, role in [(test, "test drive")] + [(e, "extra path") for e in im.EXTRA]:
            for name in ("network", "network+filter"):
                filt = sfm.SafetyFilter(m["a"], m["b"], im.TAU)
                ctrl = pol if name == "network" else sfm.SafeController(pol, filt)
                res, log = im.evaluate(ctrl if name == "network" else _with_times(ctrl), drive, cal, name)
                res.update(fold=fold, role=role)
                if name == "network+filter":
                    res.update(filt.summary())
                rows.append(res)
                print(f"   {fold} {role:10s} {drive} {name:15s}: completed {res['completed']}, row max {res['row_max_ey']}, "
                      f"turn max {res['turn_max_ey']}, corridor {res['corridor_violation_pct']} %"
                      + f", rate viol {res['rate_violation_pct']} %"
                      + (f", filter QP {res['filter_active_pct']} %, modified {res['filter_modified_pct']} %, slack {res['slack_steps']}"
                         if name != "network" else ""),
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    f = df[df.controller == "network+filter"]
    n = df[df.controller == "network"]
    check("4 full paths with the filter",
          bool(f.completed.all()) and float(f.corridor_violation_pct.max()) == 0.0 and float(f.rate_violation_pct.max()) == 0.0,
          f"{int(f.completed.sum())}/{len(f)} completed, worst corridor violation {f.corridor_violation_pct.max()} %, "
          f"rate-limit violations network alone up to {n.rate_violation_pct.max()} % -> with filter {f.rate_violation_pct.max()} %, "
          f"command modified in {f.filter_modified_pct.min()}-{f.filter_modified_pct.max()} % of steps, "
          f"total time per step median {f.infer_ms_median.median():.3f} ms (filter alone {f.filter_ms_median.median():.3f} ms)")


def _with_times(ctrl):
    """im.evaluate reads .times and calls .reset(); SafeController has both."""
    return ctrl


def plot_pair(raw, safe, P, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(raw.x, raw.y, color="tab:red", lw=1.2, label="without filter")
    ax[0].plot(safe.x, safe.y, color="tab:blue", lw=1.2, label="with filter")
    sl = (P.s > raw.s.min() - 2) & (P.s < max(raw.s.max(), safe.s.max()) + 2)
    ax[0].plot(P.x[sl], P.y[sl], color="0.7", lw=3, zorder=0, label="reference")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(raw.t, raw.e_y, color="tab:red", label="without filter")
    ax[1].plot(safe.t, safe.e_y, color="tab:blue", label="with filter")
    ax[1].axhline(1.0, color="r", ls="--", lw=0.8); ax[1].axhline(-1.0, color="r", ls="--", lw=0.8, label="corridor ±d")
    ax[1].set(xlabel="time (s)", ylabel="e_y (m)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_edge(logs, P, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(logs), figsize=(13, 4))
    for a, (off, raw, safe) in zip(ax, logs):
        a.plot(raw.t, raw.e_y, color="tab:red", label="network alone")
        a.plot(safe.t, safe.e_y, color="tab:blue", label="network + filter")
        a.axhline(1.0, color="r", ls="--", lw=0.8); a.axhline(-1.0, color="r", ls="--", lw=0.8)
        a.set(title=f"start {off} m off the path, heading outwards", xlabel="time (s)", ylabel="e_y (m)")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tests 1-3 only")
    ap.add_argument("--out", default="results/filter")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cal = sim.calibration_table()
    test_safe(cal)
    test_bad(cal, args.out)
    test_edge(cal, args.out)
    if not args.smoke:
        test_full(cal, args.out)
    print(f"\n{sum(results)}/{len(results)} tests passed")
