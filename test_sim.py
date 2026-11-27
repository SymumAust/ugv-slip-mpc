#!/usr/bin/env python3
"""
Step 2 checks: four tests of sim.py. Each prints PASS/FAIL with the number it checked.
    python test_sim.py
Writes results/sim/pure_pursuit.png and results/sim/pure_pursuit_summary.csv.
"""
import os

import numpy as np
import pandas as pd

import sim

OUT = "results/sim"
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def test_straight():
    """1 m/s command for 10 s from rest on a straight line: distance = a (10 - tau (1 - e^(-10/tau)))."""
    a, tau = 0.885, 0.2
    R = sim.Robot(a=a, b=0.98, tau=tau)
    X = np.zeros(5)
    for _ in range(100):
        X = R.step(X, np.array([1.0, 0.0]))
    expected = a * (10 - tau * (1 - np.exp(-10 / tau)))
    check("1 straight driving", abs(X[0] - expected) < 1e-3 and abs(X[1]) < 1e-9,
          f"travelled {X[0]:.3f} m, expected {expected:.3f} m (not 10 m: speed scale a = {a})")


def test_lag():
    """Step from 0 to 1 m/s: after tau seconds the speed reaches 1 - 1/e = 63.2 %."""
    tau = 0.2
    R = sim.Robot(a=0.885, b=0.98, tau=tau)
    X = R.step(np.zeros(5), np.array([1.0, 0.0]), dt=tau)
    check("2 actuator lag", abs(X[3] - (1 - np.exp(-1))) < 1e-3,
          f"speed after tau = {tau} s is {X[3]:.4f} m/s, expected {1 - np.exp(-1):.4f}")


def test_frame():
    """A pose placed exactly 0.5 m left (and right) of the path must read e_y = +-0.5, e_psi = 0,
    at 200 random places on every path, rows and turns alike."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for d in sim.ALL_DRIVES:
        P = sim.Path(d)
        for s in rng.uniform(5, P.length - 5, 200):
            for lat in (0.5, -0.5):
                info = P.frame(P.offset_pose(s, lat), s_prev=s)
                worst = max(worst, abs(info["e_y"] - lat), abs(info["e_psi"]), abs(info["s"] - s))
    check("3 path-frame errors", worst < 0.02,
          f"largest error in e_y, e_psi or s over 2000 poses on 5 paths: {worst:.4f}")


def test_pure_pursuit():
    """Drive every whole path with pure pursuit on the calibrated plant; it must reach the end."""
    cal = sim.calibration_table()
    rows, logs = [], {}
    for d in sim.ALL_DRIVES:
        P = sim.Path(d)
        p = cal["per_drive"].get(d, cal["all"])      # drives without IMU use the all-drive calibration
        log = sim.run(P, sim.Robot(p["a"], p["b"], tau=0.2), sim.pure_pursuit)
        logs[d] = log
        r, t = log[log.is_row], log[~log.is_row]
        out = (log.e_y.abs() > log.d).mean()
        rows.append(dict(drive=d, length_m=round(P.length), time_s=round(log.t.iloc[-1]),
                         reached_end=bool(log.s.iloc[-1] >= P.length - 0.5),
                         row_rms_ey_m=round(float(np.sqrt((r.e_y ** 2).mean())), 3),
                         row_max_ey_m=round(float(r.e_y.abs().max()), 3),
                         turn_max_ey_m=round(float(t.e_y.abs().max()), 3),
                         corridor_violation_pct=round(100 * float(out), 2)))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "pure_pursuit_summary.csv"), index=False)
    print(df.to_string(index=False))
    check("4 pure pursuit drives every path to the end", bool(df.reached_end.all()),
          f"{int(df.reached_end.sum())}/{len(df)} paths completed")
    figure(logs["04_13D_Jackal"], sim.Path("04_13D_Jackal"))


def figure(log, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 7))
    a1 = fig.add_subplot(1, 2, 1)
    a1.plot(P.x, P.y, color="0.75", lw=3, label="reference path")
    a1.plot(log.x, log.y, color="tab:blue", lw=0.9, label="simulated robot (pure pursuit)")
    a1.set_aspect("equal"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.set_title(f"{P.drive}: pure pursuit on the calibrated plant")
    a2 = fig.add_subplot(2, 2, 2)
    a2.plot(log.t, log.e_y, color="k", lw=0.7, label="lateral error e_y")
    a2.plot(log.t, log.d, color="tab:red", lw=1, ls="--", label="corridor +-d")
    a2.plot(log.t, -log.d, color="tab:red", lw=1, ls="--")
    a2.set(ylabel="e_y (m)", ylim=(-2.5, 2.5)); a2.legend(fontsize=8, loc="upper right"); a2.grid(alpha=0.3)
    a3 = fig.add_subplot(2, 2, 4, sharex=a2)
    a3.plot(log.t, log.v_c, lw=0.8, label="v_c (m/s)")
    a3.plot(log.t, log.w_c, lw=0.8, label="w_c (rad/s)")
    a3.set(xlabel="time (s)"); a3.legend(fontsize=8, loc="upper right"); a3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pure_pursuit.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    test_straight()
    test_lag()
    test_frame()
    test_pure_pursuit()
    print(f"\n{sum(results)}/{len(results)} tests passed")
