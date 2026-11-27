#!/usr/bin/env python3
"""
Step 4: behaviour cloning of the MPC, per leave-one-drive-out fold.

For each fold (test drive T in 02, 04, 05):
  collect  run the MPC (model fitted on the fold's training drives) over the full paths of the two
           training drives, each with its own calibrated robot; record (features, MPC command) every 0.1 s
  train    fit the policy network on those samples only
  evaluate the network drives the test drive T (robot = T's own calibration) and the extra paths 01, 03
           (robot = all-drive calibration); compared with the MPC on the same drives (step 3 results)

Plain behaviour cloning on purpose: the data are only the MPC's own trajectories, no perturbed starts,
so that step 5 (DAgger) measures exactly what on-policy data adds.

    python imitation.py              # collect (cached), train, evaluate
Outputs: data/imitation/<fold>/mpc_<drive>.npz, models/bc_<fold>.json,
         results/imitation/summary.csv, results/imitation/*.png
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import mpc
import policy
import sim

TAU = 0.2
ABORT_EY = 3.0
EXTRA = ("01_13B_Jackal", "03_13B_Jackal")
DATA, MODELS, OUT = "data/imitation", "models", "results/imitation"


def fold_name(test):
    return f"test_{test}"


def train_drives(test):
    return [d for d in sim.CALIB_DRIVES if d != test]


def plant(drive, cal):
    p = cal["per_drive"].get(drive, cal["all"])
    return sim.Robot(p["a"], p["b"], TAU)


def controller_model(test, cal):
    return cal["per_fold"][fold_name(test)]


def collect_one(job):
    test, drive = job
    cal = sim.calibration_table()
    path = os.path.join(DATA, fold_name(test), f"mpc_{drive}.npz")
    if os.path.exists(path):
        return path
    m = controller_model(test, cal)
    F, U, log = policy.record_mpc(sim.Path(drive), plant(drive, cal), mpc.MPC(m["a"], m["b"], TAU))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, F=F, U=U)
    print(f"  collected {len(F)} MPC samples on {drive} for fold {fold_name(test)}", flush=True)
    return path


def load_data(test, prefix="mpc"):
    Fs, Us = [], []
    for d in train_drives(test):
        z = np.load(os.path.join(DATA, fold_name(test), f"{prefix}_{d}.npz"))
        Fs.append(z["F"]); Us.append(z["U"])
    return np.vstack(Fs), np.vstack(Us)


def rate_violations(log):
    """Share of steps where the applied command changed faster than the data-based rate limits
    (|dv_c| > acc_max dt or |dw_c| > alpha_max dt): an input-constraint violation."""
    L = sim.LIMITS
    lim = np.array([L["acc_max"], L["alpha_max"]]) * sim.DT_CTRL
    d = np.abs(np.diff(log[["v_c", "w_c"]].to_numpy(), axis=0))
    return float(np.mean(np.any(d > lim + 1e-6, axis=1))) if len(d) else 0.0


def evaluate(pol, drive, cal, label):
    P = sim.Path(drive)
    pol.reset()
    log = sim.run(P, plant(drive, cal), pol, abort_ey=ABORT_EY)
    aborted = bool(log.get("aborted", pd.Series(False)).fillna(False).any())
    r, t = log[log.is_row], log[~log.is_row]
    ms = np.array(pol.times) * 1e3
    return dict(controller=label, drive=drive, completed=bool(log.s.iloc[-1] >= P.length - 0.5) and not aborted,
                aborted=aborted, progress_pct=round(100 * float(log.s.iloc[-1] / P.length), 1),
                time_s=round(float(log.t.iloc[-1]), 1),
                row_rms_ey=round(float(np.sqrt((r.e_y ** 2).mean())), 4) if len(r) else np.nan,
                row_max_ey=round(float(r.e_y.abs().max()), 3) if len(r) else np.nan,
                turn_rms_ey=round(float(np.sqrt((t.e_y ** 2).mean())), 4) if len(t) else np.nan,
                turn_max_ey=round(float(t.e_y.abs().max()), 3) if len(t) else np.nan,
                corridor_violation_pct=round(100 * float((log.e_y.abs() > log.d).mean()), 3),
                rate_violation_pct=round(100 * rate_violations(log), 3),
                infer_ms_median=round(float(np.median(ms)), 4), infer_ms_p99=round(float(np.percentile(ms, 99)), 4)), log


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    cal = sim.calibration_table()
    jobs = [(t, d) for t in sim.CALIB_DRIVES for d in train_drives(t)]
    print(f"collecting MPC data: {len(jobs)} runs (cached runs are skipped)", flush=True)
    with Pool(min(2, os.cpu_count() or 1)) as pool:
        pool.map(collect_one, jobs)

    rows, logs = [], {}
    for test in sim.CALIB_DRIVES:
        F, U = load_data(test)
        pol = policy.Policy.fit(F, U)
        pol.save(os.path.join(MODELS, f"bc_{fold_name(test)}.json"))
        print(f"fold {fold_name(test)}: trained on {pol.train_info['n_samples']} samples from {train_drives(test)}, "
              f"{pol.train_info['epochs']} epochs, validation R^2 {pol.train_info['best_val_r2']:.4f}", flush=True)
        for drive, role in [(test, "test drive")] + [(e, "extra path") for e in EXTRA]:
            res, log = evaluate(pol, drive, cal, "bc")
            res.update(fold=fold_name(test), role=role, n_train=pol.train_info["n_samples"],
                       val_r2=round(pol.train_info["best_val_r2"], 4))
            rows.append(res)
            logs[(test, drive)] = log
            print(f"   {role:10s} {drive}: completed {res['completed']} ({res['progress_pct']} %), "
                  f"row max |e_y| {res['row_max_ey']}, turn max |e_y| {res['turn_max_ey']}, "
                  f"corridor violations {res['corridor_violation_pct']} %, inference {res['infer_ms_median']} ms", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    compare(df)
    figures(logs, df)


def compare(df):
    """Side by side with the MPC on the same drives (step 3, results/mpc/summary.csv)."""
    m = pd.read_csv("results/mpc/summary.csv")
    m = m[m.controller == "mpc"].set_index("drive")
    out = []
    for _, r in df.iterrows():
        mm = m.loc[r.drive]
        out.append(dict(fold=r.fold, drive=r.drive, role=r.role, bc_completed=r.completed,
                        mpc_row_max=mm.row_max_ey, bc_row_max=r.row_max_ey,
                        mpc_turn_max=mm.turn_max_ey, bc_turn_max=r.turn_max_ey,
                        mpc_turn_rms=mm.turn_rms_ey, bc_turn_rms=r.turn_rms_ey,
                        mpc_time=mm.time_s, bc_time=r.time_s,
                        mpc_ms_median=mm.solve_ms_median, bc_ms_median=r.infer_ms_median))
    c = pd.DataFrame(out)
    c.to_csv(os.path.join(OUT, "bc_vs_mpc.csv"), index=False)
    print("\nBehaviour cloning vs MPC (max |e_y| in m, time in s, compute in ms per step):")
    print(c.to_string(index=False))


def figures(logs, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = "04_13D_Jackal"
    log = logs[(test, test)]
    P = sim.Path(test)
    fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    ax[0].plot(log.s, log.v_c, lw=0.8, color="tab:green", label="behaviour cloning")
    ax[1].plot(log.s, log.e_y, lw=0.8, color="tab:green")
    ax[1].plot(P.s, P.d, "r--", lw=0.8); ax[1].plot(P.s, -P.d, "r--", lw=0.8, label="corridor ±d")
    for a in ax:
        a.fill_between(P.s, -5, 5, where=~P.is_row, color="tab:red", alpha=0.08, step="mid")
    ax[0].set(ylabel="v_c (m/s)", ylim=(0, 1.9), title=f"Behaviour-cloned network driving {test} (unseen test drive; turns shaded)")
    lim = max(0.3, 1.2 * float(log.e_y.abs().max()))
    ax[1].set(ylabel="e_y (m)", xlabel="s (m)", ylim=(-min(lim, 2.3), min(lim, 2.3)))
    ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bc_drive04.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
