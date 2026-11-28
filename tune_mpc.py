#!/usr/bin/env python3
"""
Step 7a: systematic tuning of the MPC weights (a stronger teacher and a fairer baseline).

Leave-one-drive-out, like everything else: for the fold that tests drive T, the weights are chosen
only from runs on the fold's two TRAINING drives (plant = that drive's own a, b; MPC model = the
fold's calibration). The test drive and the extra paths 01, 03 are never used for the choice.

Grid (36):  q_y in {10, 20, 40, 80}, q_psi in {1, 3, 10}, r_dw in {2, 8, 32};
            terminal weights qN_y = 2 q_y, qN_psi = 2 q_psi; q_v, r_dv, slack penalties and v_ref unchanged.
            The current weights (20, 3, 8) are part of the grid.

Selection rule (fixed before any run):
  admissible  both training drives completed, zero corridor violations, zero solver failures,
              time <= 1.03 x the current weights' time on the same drive (no buying accuracy with speed)
  score       worst max |e_y| (rows and turns) over the two training drives -> smaller is better
  tie-break   within 0.005 m of the best score: smallest mean (row RMS + turn RMS)

Then the chosen weights and the current weights are both evaluated on each fold's test drive and the
extra paths (same plants and models as the networks are evaluated with).

    python tune_mpc.py             # full grid (resumable: finished runs are kept in grid_runs.csv)
    python tune_mpc.py --smoke     # 2 grid points, one fold, short runs, output in /tmp
Outputs: results/mpc_tuning/grid_runs.csv, grid_scores.csv, test.csv, config/mpc_weights.json
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import imitation
import mpc
import sim

TAU = 0.2
GRID = dict(q_y=[10.0, 20.0, 40.0, 80.0], q_psi=[1.0, 3.0, 10.0], r_dw=[2.0, 8.0, 32.0])
CURRENT = dict(q_y=mpc.WEIGHTS["q_y"], q_psi=mpc.WEIGHTS["q_psi"], r_dw=mpc.WEIGHTS["r_dw"])
TIME_FACTOR = 1.03
TIE_M = 0.005
OUT = "results/mpc_tuning"
SMOKE_T_MAX = None


def full_weights(g):
    return dict(q_y=g["q_y"], q_psi=g["q_psi"], r_dw=g["r_dw"], qN_y=2 * g["q_y"], qN_psi=2 * g["q_psi"])


def key(g):
    return f"qy{g['q_y']:g}_qpsi{g['q_psi']:g}_rdw{g['r_dw']:g}"


def run_mpc(drive, model, weights, t_max=None):
    cal = sim.calibration_table()
    P = sim.Path(drive)
    ctrl = mpc.MPC(model["a"], model["b"], TAU, weights=weights)
    log = sim.run(P, imitation.plant(drive, cal), ctrl, t_max=t_max, abort_ey=imitation.ABORT_EY)
    aborted = bool(log.get("aborted", pd.Series(False)).fillna(False).any())
    r, t = log[log.is_row], log[~log.is_row]
    sm = ctrl.summary()
    full = t_max is None
    return dict(drive=drive,
                completed=(bool(log.s.iloc[-1] >= P.length - 0.5) or not full) and not aborted,
                time_s=round(float(log.t.iloc[-1]), 1),
                row_rms_ey=round(float(np.sqrt((r.e_y ** 2).mean())), 4) if len(r) else np.nan,
                row_max_ey=round(float(r.e_y.abs().max()), 3) if len(r) else np.nan,
                turn_rms_ey=round(float(np.sqrt((t.e_y ** 2).mean())), 4) if len(t) else np.nan,
                turn_max_ey=round(float(t.e_y.abs().max()), 3) if len(t) else np.nan,
                corridor_violation_pct=round(100 * float((log.e_y.abs() > log.d).mean()), 3),
                rate_violation_pct=round(100 * imitation.rate_violations(log), 3),
                failures=sm["failures"], slack_steps=sm["slack_steps"],
                solve_ms_median=sm["solve_ms_median"], solve_ms_p99=sm["solve_ms_p99"])


def grid_job(job):
    test, drive, g = job
    cal = sim.calibration_table()
    res = run_mpc(drive, imitation.controller_model(test, cal), full_weights(g), t_max=SMOKE_T_MAX)
    res.update(fold=imitation.fold_name(test), key=key(g), **g)
    return res


def score_fold(runs, fold):
    """Apply the selection rule to the grid runs of one fold. Returns the scored table."""
    f = runs[runs.fold == fold].copy()
    f["max_ey"] = f[["row_max_ey", "turn_max_ey"]].max(axis=1)
    f["rms_sum"] = f.row_rms_ey + f.turn_rms_ey
    cur = f[f.key == key(CURRENT)].set_index("drive").time_s
    f["time_ok"] = f.apply(lambda r: r.time_s <= TIME_FACTOR * cur[r.drive], axis=1)
    f["ok"] = f.completed & (f.corridor_violation_pct == 0) & (f.failures == 0) & f.time_ok
    s = f.groupby("key").agg(q_y=("q_y", "first"), q_psi=("q_psi", "first"), r_dw=("r_dw", "first"),
                             n=("drive", "size"), admissible=("ok", "all"), score=("max_ey", "max"),
                             rms=("rms_sum", "mean"), time_s=("time_s", "sum"),
                             solve_ms_median=("solve_ms_median", "median")).reset_index()
    s["admissible"] &= s.n == len(imitation.train_drives(fold[5:]))
    s["fold"] = fold
    adm = s[s.admissible]
    best = adm.score.min()
    near = adm[adm.score <= best + TIE_M]
    chosen = near.sort_values("rms").iloc[0].key
    s["chosen"] = s.key == chosen
    s["current"] = s.key == key(CURRENT)
    return s.sort_values(["admissible", "score"], ascending=[False, True])


def test_job(job):
    test, drive, role, label, g = job
    cal = sim.calibration_table()
    res = run_mpc(drive, imitation.controller_model(test, cal), full_weights(g), t_max=SMOKE_T_MAX)
    res.update(fold=imitation.fold_name(test), role=role, controller=label, key=key(g))
    return res


def main():
    global SMOKE_T_MAX, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = ap.parse_args()
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    folds = list(sim.CALIB_DRIVES)
    if args.smoke:
        SMOKE_T_MAX, OUT = 40.0, "/tmp/mpc_tuning_smoke"
        grid = [CURRENT, dict(q_y=40.0, q_psi=3.0, r_dw=8.0)]
        folds = folds[:1]
    os.makedirs(OUT, exist_ok=True)

    # ---------------------------------------------------------------- grid on the training drives
    runs_path = os.path.join(OUT, "grid_runs.csv")
    done = pd.read_csv(runs_path) if os.path.exists(runs_path) else pd.DataFrame()
    done_keys = set(zip(done.fold, done.drive, done.key)) if len(done) else set()
    jobs = [(t, d, g) for t in folds for d in imitation.train_drives(t) for g in grid
            if (imitation.fold_name(t), d, key(g)) not in done_keys]
    print(f"grid: {len(grid)} weight sets x {len(folds)} folds x 2 training drives; "
          f"{len(jobs)} runs to do ({len(done_keys)} already done), {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(grid_job, jobs), 1):
            pd.DataFrame([res]).to_csv(runs_path, mode="a", header=not os.path.exists(runs_path), index=False)
            print(f"  [{i}/{len(jobs)}] {res['fold'][5:7]} {res['drive'][:2]} {res['key']:<22} "
                  f"max {max(res['row_max_ey'], res['turn_max_ey']):.3f} m  time {res['time_s']:.0f} s  "
                  f"corr {res['corridor_violation_pct']}%  fail {res['failures']}  "
                  f"{res['solve_ms_median']:.1f} ms", flush=True)
    runs = pd.read_csv(runs_path)

    scores = pd.concat([score_fold(runs, imitation.fold_name(t)) for t in folds])
    scores.to_csv(os.path.join(OUT, "grid_scores.csv"), index=False)
    chosen = {}
    for t in folds:
        fn = imitation.fold_name(t)
        s = scores[scores.fold == fn]
        c, cur = s[s.chosen].iloc[0], s[s.current].iloc[0]
        chosen[fn] = full_weights(dict(q_y=c.q_y, q_psi=c.q_psi, r_dw=c.r_dw))
        print(f"\nfold {fn}: {int(s.admissible.sum())}/{len(s)} weight sets admissible", flush=True)
        print(f"  current {cur.key:<22} worst max {cur.score:.3f} m  mean RMS sum {cur.rms:.4f}")
        print(f"  chosen  {c.key:<22} worst max {c.score:.3f} m  mean RMS sum {c.rms:.4f}")
        print(s.head(5)[["key", "admissible", "score", "rms", "time_s", "solve_ms_median"]].to_string(index=False))
    wpath = os.path.join("config", "mpc_weights.json") if not args.smoke else os.path.join(OUT, "mpc_weights.json")
    json.dump(dict(per_fold=chosen, rule=f"min worst max|e_y| on the fold's training drives; time <= {TIME_FACTOR} x "
                   f"current; tie within {TIE_M} m -> min RMS", grid=GRID), open(wpath, "w"), indent=2)
    print(f"\nwrote {wpath}", flush=True)

    # ---------------------------------------------------------------- held-out evaluation
    jobs = []
    for t in folds:
        c = scores[(scores.fold == imitation.fold_name(t)) & scores.chosen].iloc[0]
        g = dict(q_y=c.q_y, q_psi=c.q_psi, r_dw=c.r_dw)
        for drive, role in [(t, "test drive")] + [(e, "extra path") for e in imitation.EXTRA]:
            jobs.append((t, drive, role, "mpc_current", CURRENT))
            jobs.append((t, drive, role, "mpc_tuned", g))
    print(f"held-out evaluation: {len(jobs)} runs", flush=True)
    with Pool(args.workers) as pool:
        test = pd.DataFrame(pool.map(test_job, jobs))
    test.to_csv(os.path.join(OUT, "test.csv"), index=False)
    cols = ["fold", "drive", "controller", "key", "completed", "time_s", "row_rms_ey", "row_max_ey",
            "turn_rms_ey", "turn_max_ey", "corridor_violation_pct", "slack_steps", "solve_ms_median", "solve_ms_p99"]
    t2 = test[cols].copy()
    t2["fold"], t2["drive"] = t2.fold.str[5:7], t2.drive.str[:2]
    print(t2.sort_values(["fold", "drive", "controller"]).to_string(index=False))


if __name__ == "__main__":
    main()
