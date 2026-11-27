#!/usr/bin/env python3
"""
Step 5: DAgger (Ross et al., 2011) on top of the behaviour-cloned policy of step 4.

Per fold (test drive T; training drives = the other two calibrated drives), round i = 1 ... ROUNDS:
  1. the CURRENT network drives each training path (its own calibrated robot), mistakes included;
  2. at every state it visits, the MPC (the fold's controller model) is asked what it would command.
     The network's command is applied, the MPC's answer is only recorded. The MPC is told the
     network's last applied command as its "previous command", so its answer respects the same
     rate limits from the state the robot is really in;
  3. those (state, MPC answer) pairs are added to all earlier data (BC data + earlier rounds);
  4. a new network is trained from scratch on the aggregate and evaluated on the test drive and the
     extra paths 01, 03. The test drive is never used for training or labelling.

Round 0 = the step-4 behaviour-cloning network.
    python dagger.py
Outputs: data/dagger/<fold>/r<i>_<drive>.npz, models/dagger_<fold>_r<i>.json,
         results/dagger/rounds.csv, results/dagger/*.png
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import imitation as im
import mpc
import policy
import sim

ROUNDS = 5
T_MAX = 900.0          # cap on one DAgger rollout (s of simulated time); a slow policy still visits
                       # plenty of informative states, and the labelling cost stays bounded
DATA, OUT = "data/dagger", "results/dagger"


def dagger_rollout(pol, drive, test, cal):
    """Network drives `drive`; the MPC labels every visited state. Returns features, labels, log."""
    m = im.controller_model(test, cal)
    expert = mpc.MPC(m["a"], m["b"], im.TAU)
    P = sim.Path(drive)
    feats, labels = [], []
    pol.reset()

    def ctrl(P_, X, info):
        f = policy.features(X, info, pol.u_prev.copy())
        expert.u_prev = pol.u_prev.copy()          # the state the robot is really in
        labels.append(expert(P_, X, info).copy())  # recorded only
        feats.append(f)
        return pol(P_, X, info)                    # applied

    log = sim.run(P, im.plant(drive, cal), ctrl, t_max=T_MAX, abort_ey=im.ABORT_EY)
    return np.array(feats), np.array(labels), log, expert.summary()


def run_fold(test):
    cal = sim.calibration_table()
    fold = im.fold_name(test)
    os.makedirs(os.path.join(DATA, fold), exist_ok=True)
    F_all, U_all = im.load_data(test)                              # round 0 data = MPC's own rollouts
    pol = policy.Policy.load(os.path.join(im.MODELS, f"bc_{fold}.json"))
    rows = []

    def evaluate_round(pol, r, n_train):
        for drive, role in [(test, "test drive")] + [(e, "extra path") for e in im.EXTRA]:
            res, _ = im.evaluate(pol, drive, cal, f"dagger_r{r}")
            res.update(fold=fold, round=r, role=role, n_train=n_train)
            rows.append(res)
        t = rows[-3]
        print(f"[{fold}] round {r}: {n_train} samples | test drive: time {t['time_s']} s, row max |e_y| {t['row_max_ey']}, "
              f"turn max |e_y| {t['turn_max_ey']}, corridor viol {t['corridor_violation_pct']} %, completed {t['completed']}",
              flush=True)

    evaluate_round(pol, 0, len(F_all))
    for r in range(1, ROUNDS + 1):
        for drive in im.train_drives(test):
            path = os.path.join(DATA, fold, f"r{r}_{drive}.npz")
            if os.path.exists(path):
                z = np.load(path)
                F, U = z["F"], z["U"]
            else:
                F, U, log, es = dagger_rollout(pol, drive, test, cal)
                np.savez_compressed(path, F=F, U=U)
                print(f"[{fold}] round {r}: network drove {drive} for {log.t.iloc[-1]:.0f} s "
                      f"({100 * log.s.iloc[-1] / sim.Path(drive).length:.0f} % of the path), {len(F)} states labelled, "
                      f"MPC failures {es['failures']}", flush=True)
            F_all, U_all = np.vstack([F_all, F]), np.vstack([U_all, U])
        pol = policy.Policy.fit(F_all, U_all)
        pol.save(os.path.join(im.MODELS, f"dagger_{fold}_r{r}.json"))
        evaluate_round(pol, r, len(F_all))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    with Pool(min(2, os.cpu_count() or 1)) as pool:
        all_rows = pool.map(run_fold, list(sim.CALIB_DRIVES))
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv(os.path.join(OUT, "rounds.csv"), index=False)
    t = df[df.role == "test drive"][["fold", "round", "n_train", "completed", "time_s", "row_rms_ey", "row_max_ey",
                                     "turn_rms_ey", "turn_max_ey", "corridor_violation_pct"]]
    print("\nTest drives, round by round:")
    print(t.to_string(index=False))
    figures(df)


def figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = pd.read_csv("results/mpc/summary.csv")
    m = m[m.controller == "mpc"].set_index("drive")
    t = df[df.role == "test drive"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for fold, g in t.groupby("fold"):
        d = g.drive.iloc[0]
        lab = d[:2]
        ax[0].plot(g["round"], g.time_s / m.loc[d, "time_s"], "o-", label=f"test {lab}")
        ax[1].plot(g["round"], g.row_max_ey, "o-", label=f"test {lab}")
        ax[2].plot(g["round"], g.turn_max_ey, "o-", label=f"test {lab}")
        ax[3].plot(g["round"], g.corridor_violation_pct, "o-", label=f"test {lab}")
    ax[0].axhline(1, color="k", lw=0.8, ls=":")
    ax[0].set(title="time to finish / MPC's time", xlabel="DAgger round")
    ax[1].set(title="max |e_y| in rows (m)", xlabel="DAgger round")
    ax[2].set(title="max |e_y| in turns (m)", xlabel="DAgger round")
    ax[3].set(title="corridor violations (% of steps)", xlabel="DAgger round")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("DAgger: round 0 = behaviour cloning; each round adds MPC labels on states the network itself visited")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "rounds.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
