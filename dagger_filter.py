#!/usr/bin/env python3
"""
Step 6b: DAgger with the safety filter in the loop.

Why: the step-5 network was trained without any limit on how fast its command may change. Deployed
behind the filter, its fast corrections are slowed down (rate limits), it has never seen that, it
corrects harder, and on path 01 (fold 05) the weaving grew until the robot left the corridor.
Principle: a safety layer used at deployment must also be present during training.

Second fix, found while building this step: the step-5 network outputs the command itself. In 1.5 % of
the MPC's labels the command changes exactly at the rate limit; a tiny imitation error then pushes the
network past the limit (0.7-1.1 % of steps even on its own training states). More data cannot fix
that. So from round 1 on the network outputs the CHANGE of the command, clipped to the rate limits
(policy.Policy.fit(delta=True)): it can never break them.

Rounds (per fold):
  round 0   the step-5 network (command output), unchanged
  round 1   same data (BC + DAgger rounds 1-5), retrained with change output: effect of the output form alone
  round 2 ... 1+ROUNDS:
    1. network + filter drive each training path (the filter's command is what the robot gets);
    2. the MPC labels every visited state; its "previous command" is the command the filter really
       applied, so the labels respect the same rate limits from the real state;
    3. aggregate, retrain from scratch, evaluate on the test drive and the extra paths, both
       network alone and network + filter.
The test drive is never used for training or labelling.

    python dagger_filter.py            # full run (3 folds in parallel)
    python dagger_filter.py --smoke    # 1 short rollout on one fold, writes nothing to the repo
Outputs: data/dagger_filter/<fold>/r<i>_<drive>.npz, models/dagger_filter_<fold>_r<i>.json,
         results/dagger_filter/rounds.csv, results/dagger_filter/rounds.png
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import dagger
import imitation as im
import mpc
import policy
import safety_filter as sfm
import sim

ROUNDS = 3
T_MAX = 900.0
DATA, OUT = "data/dagger_filter", "results/dagger_filter"
CFG = dict(rounds=ROUNDS, t_max=T_MAX, data=DATA, out=OUT, models=im.MODELS, folds=list(sim.CALIB_DRIVES))


def make_filter(test, cal):
    m = im.controller_model(test, cal)
    return sfm.SafetyFilter(m["a"], m["b"], im.TAU)


def filtered_rollout(pol, drive, test, cal, t_max):
    """Network + filter drive `drive`; the MPC labels every visited state from the applied command."""
    m = im.controller_model(test, cal)
    expert = mpc.MPC(m["a"], m["b"], im.TAU)
    filt = make_filter(test, cal)
    safe = sfm.SafeController(pol, filt)
    safe.reset()
    feats, labels = [], []

    def ctrl(P_, X, info):
        f = policy.features(X, info, safe.u_prev.copy())   # what the network sees = the applied command
        expert.u_prev = safe.u_prev.copy()
        labels.append(expert(P_, X, info).copy())          # recorded only
        feats.append(f)
        return safe(P_, X, info)                           # network -> filter -> robot

    log = sim.run(sim.Path(drive), im.plant(drive, cal), ctrl, t_max=t_max, abort_ey=im.ABORT_EY)
    return np.array(feats), np.array(labels), log, filt.summary()


def all_previous_data(test):
    """BC data + the five step-5 DAgger rounds of this fold."""
    F, U = im.load_data(test)
    fold = im.fold_name(test)
    for r in range(1, dagger.ROUNDS + 1):
        for d in im.train_drives(test):
            z = np.load(os.path.join(dagger.DATA, fold, f"r{r}_{d}.npz"))
            F, U = np.vstack([F, z["F"]]), np.vstack([U, z["U"]])
    return F, U


def evaluate_both(pol, test, cal, r, n_train, rows, drives):
    fold = im.fold_name(test)
    for drive, role in drives:
        for name in ("network", "network+filter"):
            filt = make_filter(test, cal)
            ctrl = pol if name == "network" else sfm.SafeController(pol, filt)
            res, _ = im.evaluate(ctrl, drive, cal, name)
            res.update(fold=fold, round=r, role=role, n_train=n_train)
            if name != "network":
                res.update(filt.summary())
            rows.append(res)
            print(f"[{fold}] round {r} {role:10s} {drive} {name:15s}: row max {res['row_max_ey']}, turn max {res['turn_max_ey']}, "
                  f"corridor {res['corridor_violation_pct']} %, rate viol {res['rate_violation_pct']} %"
                  + (f", modified {res['filter_modified_pct']} %, slack {res['slack_steps']}" if name != "network" else ""),
                  flush=True)


def run_fold(args):
    test, cfg = args
    cal = sim.calibration_table()
    fold = im.fold_name(test)
    os.makedirs(os.path.join(cfg["data"], fold), exist_ok=True)
    F_all, U_all = all_previous_data(test)
    pol = policy.Policy.load(os.path.join(im.MODELS, f"dagger_{fold}_r{dagger.ROUNDS}.json"))
    drives = [(test, "test drive")] + [(e, "extra path") for e in im.EXTRA]
    if cfg.get("smoke"):
        drives = [(test, "test drive")]
    rows = []
    evaluate_both(pol, test, cal, 0, len(F_all), rows, drives)       # round 0 = the step-5 network
    pol = policy.Policy.fit(F_all, U_all, delta=True)                # round 1 = change output, same data
    pol.save(os.path.join(cfg["models"], f"dagger_filter_{fold}_r1.json"))
    evaluate_both(pol, test, cal, 1, len(F_all), rows, drives)
    for r in range(2, cfg["rounds"] + 2):
        for drive in im.train_drives(test):
            path = os.path.join(cfg["data"], fold, f"r{r}_{drive}.npz")
            if os.path.exists(path):
                z = np.load(path); F, U = z["F"], z["U"]
            else:
                F, U, log, fs = filtered_rollout(pol, drive, test, cal, cfg["t_max"])
                np.savez_compressed(path, F=F, U=U)
                print(f"[{fold}] round {r}: network+filter drove {drive} for {log.t.iloc[-1]:.0f} s "
                      f"({100 * log.s.iloc[-1] / sim.Path(drive).length:.0f} % of the path), {len(F)} states labelled, "
                      f"filter modified {fs['filter_modified_pct']} % of commands", flush=True)
            F_all, U_all = np.vstack([F_all, F]), np.vstack([U_all, U])
        pol = policy.Policy.fit(F_all, U_all, delta=True)
        pol.save(os.path.join(cfg["models"], f"dagger_filter_{fold}_r{r}.json"))
        evaluate_both(pol, test, cal, r, len(F_all), rows, drives)
    return rows


def figures(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for (fold, name), g in df.groupby(["fold", "controller"]):
        g = g.groupby("round").agg(row=("row_max_ey", "max"), turn=("turn_max_ey", "max"),
                                    corr=("corridor_violation_pct", "max"), rate=("rate_violation_pct", "max"))
        ls = "-" if name == "network+filter" else ":"
        lab = f"test {fold[5:7]} {name}"
        for a, col in zip(ax, ("row", "turn", "rate", "corr")):    # g["corr"], not g.corr (a pandas method)
            a.plot(g.index, g[col], linestyle=ls, marker="o", label=lab)
    for a, t in zip(ax, ["worst max |e_y| in rows (m)", "worst max |e_y| in turns (m)",
                         "worst rate-limit violations (%)", "worst corridor violations (%)"]):
        a.set(title=t, xlabel="filter-in-the-loop round"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("Round 0 = step-5 network; 1 = change output (same data); 2-4 = + filter-in-the-loop DAgger. "
                 "Worst over test drive and extra paths; solid = with filter, dotted = network alone")
    fig.tight_layout(); fig.savefig(os.path.join(out, "rounds.png"), dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.smoke:
        cfg.update(rounds=1, t_max=20.0, data="/tmp/dagger_filter_smoke", out="/tmp/dagger_filter_smoke_out",
                   models="/tmp/dagger_filter_smoke_models", folds=["05_13D_Jackal"], smoke=True)
        os.makedirs(cfg["models"], exist_ok=True)
    os.makedirs(cfg["out"], exist_ok=True)
    with Pool(min(3, os.cpu_count() or 1)) as pool:
        all_rows = pool.map(run_fold, [(t, cfg) for t in cfg["folds"]])
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv(os.path.join(cfg["out"], "rounds.csv"), index=False)
    cols = ["fold", "round", "drive", "controller", "row_max_ey", "turn_max_ey", "corridor_violation_pct",
            "rate_violation_pct", "time_s"]
    print("\nAll results:")
    print(df[cols].to_string(index=False))
    figures(df, cfg["out"])


if __name__ == "__main__":
    main()
