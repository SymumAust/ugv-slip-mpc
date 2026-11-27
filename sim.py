#!/usr/bin/env python3
"""
Step 2: simulator for the orchard path-tracking study.

Three parts:
  Robot      the plant of the proposal, eq. (1):
                 x' = a v cos(psi),  y' = a v sin(psi),  psi' = b w,
                 tau v' = v_c - v,   tau w' = w_c - w
             state X = [x, y, psi, v, w] (v, w = wheel-odometry speed and turn rate),
             input U = [v_c, w_c] (commanded wheel speed and turn rate).
             The controller acts every DT_CTRL = 0.1 s; inside, the robot is integrated with
             RK4 in DT_SIM = 0.02 s steps.
  Path       a reference path from step 1 (data/paths/<drive>.csv) and the path-frame
             quantities every controller sees: progress s, lateral error e_y, heading error
             e_psi, curvature preview kappa at fixed distances ahead, corridor half-width d.
  calibrate  the plant parameters a, b fitted from the real drives (feasibility-check method),
             per drive and per leave-one-drive-out fold.
"""
import json
import os

import numpy as np
import pandas as pd

DT_CTRL = 0.1                                   # controller period (s)
DT_SIM = 0.02                                   # integration step (s)
PREVIEW_DIST = 0.4 * np.arange(1, 9)            # curvature preview: 0.4 ... 3.2 m ahead
                                                # (3.2 m = 2 s at the top true speed of 1.6 m/s)
LIMITS = json.load(open(os.path.join(os.path.dirname(__file__), "config", "limits.json")))
CORRIDOR = json.load(open(os.path.join(os.path.dirname(__file__), "config", "corridor.json")))
CALIB_DRIVES = ("02_13B_Jackal", "04_13D_Jackal", "05_13D_Jackal")   # drives with IMU data
ALL_DRIVES = ("01_13B_Jackal", "02_13B_Jackal", "03_13B_Jackal", "04_13D_Jackal", "05_13D_Jackal")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------------------- robot
class Robot:
    def __init__(self, a=0.885, b=0.98, tau=0.2):
        self.a, self.b, self.tau = a, b, tau

    def f(self, X, U):
        x, y, psi, v, w = X
        return np.array([self.a * v * np.cos(psi), self.a * v * np.sin(psi), self.b * w,
                         (U[0] - v) / self.tau, (U[1] - w) / self.tau])

    def step(self, X, U, dt=DT_CTRL):
        """Advance by dt (one controller period) with RK4 sub-steps; U is held constant."""
        U = clip_command(U)
        n = int(round(dt / DT_SIM))
        h = dt / n
        for _ in range(n):
            k1 = self.f(X, U)
            k2 = self.f(X + h / 2 * k1, U)
            k3 = self.f(X + h / 2 * k2, U)
            k4 = self.f(X + h * k3, U)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X


def clip_command(U):
    return np.array([np.clip(U[0], LIMITS["v_c_min"], LIMITS["v_c_max"]),
                     np.clip(U[1], -LIMITS["w_c_max"], LIMITS["w_c_max"])])


# ------------------------------------------------------------------------------------------ path
class Path:
    """Reference path with path-frame projection. Paths revisit lanes, so the projection searches
    only near the previous progress s (a local window), never the whole path."""

    def __init__(self, drive, d_row=None, d_turn=None):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "paths", f"{drive}.csv"))
        self.drive = drive
        self.s, self.x, self.y = df.s.to_numpy(), df.x.to_numpy(), df.y.to_numpy()
        self.psi, self.kappa = df.psi.to_numpy(), df.kappa.to_numpy()
        self.is_row = (df.seg == "row").to_numpy()
        self.ds = float(np.median(np.diff(self.s)))
        self.length = float(self.s[-1])
        d_row = CORRIDOR["d_row_m"] if d_row is None else d_row
        d_turn = CORRIDOR["d_turn_m"] if d_turn is None else d_turn
        self.d = np.where(self.is_row, d_row, d_turn)

    def index(self, s):
        return int(np.clip(round(s / self.ds), 0, len(self.s) - 1))

    def interp(self, s, arr):
        return np.interp(s, self.s, arr)

    def start_state(self, v=0.0):
        return np.array([self.x[0], self.y[0], self.psi[0], v, 0.0])

    def project(self, x, y, s_prev, window=5.0):
        """Closest point to (x, y) within +-window metres of s_prev; returns s refined along the tangent."""
        lo, hi = self.index(s_prev - window), self.index(s_prev + window) + 1
        dx, dy = self.x[lo:hi] - x, self.y[lo:hi] - y
        i = lo + int(np.argmin(dx * dx + dy * dy))
        t = np.cos(self.psi[i]) * (x - self.x[i]) + np.sin(self.psi[i]) * (y - self.y[i])
        return float(np.clip(self.s[i] + t, 0.0, self.length))

    def frame(self, X, s_prev):
        """Path-frame quantities for state X: s, e_y (left of path positive), e_psi, kappa preview, d."""
        x, y, psi = X[0], X[1], X[2]
        s = self.project(x, y, s_prev)
        xr, yr = self.interp(s, self.x), self.interp(s, self.y)
        pr = self.interp(s, np.unwrap(self.psi))
        e_y = -np.sin(pr) * (x - xr) + np.cos(pr) * (y - yr)
        e_psi = wrap(psi - pr)
        kap = self.interp(np.minimum(s + PREVIEW_DIST, self.length), self.kappa)
        return dict(s=s, e_y=float(e_y), e_psi=float(e_psi), kappa=self.interp(s, self.kappa),
                    preview=kap, d=float(self.d[self.index(s)]), is_row=bool(self.is_row[self.index(s)]))

    def offset_pose(self, s, lateral):
        """Pose `lateral` metres to the left of the path at progress s, heading along the path (for tests)."""
        pr = self.interp(s, np.unwrap(self.psi))
        return np.array([self.interp(s, self.x) - lateral * np.sin(pr),
                         self.interp(s, self.y) + lateral * np.cos(pr), pr, 0.0, 0.0])


# ------------------------------------------------------------------------------------ calibration
def calibrate(drives, lever_xy=(-0.18, 0.0)):
    """Fit a (speed scale) and b (turn-rate scale) on the given drives, with the feasibility check's
    preprocessing (time sync, heading fusion, GPS lever arm) and its least-squares ICR fit."""
    import feasibility as fz
    cfg = dict(data_root="/nonexistent", processed_root=os.path.join(os.path.dirname(__file__), "data", "processed"),
               lever_arm_xy=list(lever_xy))
    F, Y = [], []
    for d in drives:
        D, Q = fz.process_drive(d, d[3:6], cfg)
        m = D["moving"]
        F.append(D["F"][m][::fz.TRAIN_DOWNSAMPLE])
        Y.append(D["Y"][m][::fz.TRAIN_DOWNSAMPLE])
    F, Y = np.vstack(F), np.vstack(Y)
    v, w = F[:, 0], F[:, 1]
    return dict(a=float(v @ Y[:, 0] / (v @ v)), b=float(w @ Y[:, 2] / (w @ w)))


def calibration_table(path=os.path.join(os.path.dirname(__file__), "config", "calibration.json")):
    """a, b per drive and per leave-one-drive-out fold (cached in config/calibration.json)."""
    if os.path.exists(path):
        return json.load(open(path))
    table = dict(per_drive={d: calibrate([d]) for d in CALIB_DRIVES},
                 per_fold={f"test_{d}": calibrate([e for e in CALIB_DRIVES if e != d]) for d in CALIB_DRIVES},
                 all=calibrate(list(CALIB_DRIVES)),
                 note="per_drive: plant of that drive when it is the test drive; per_fold[test_X]: "
                      "controller model fitted on the training drives of the fold that tests X.")
    json.dump(table, open(path, "w"), indent=2)
    return table


# ----------------------------------------------------------------------------- baseline controller
def pure_pursuit(P, X, info, lookahead=1.2, v_row=1.4, v_turn=0.8):
    """The simplest path follower: steer towards the path point `lookahead` metres ahead.
    Used only to check the simulator and as an extra, non-optimising baseline."""
    s_t = min(info["s"] + lookahead, P.length)
    xt, yt = P.interp(s_t, P.x), P.interp(s_t, P.y)
    alpha = wrap(np.arctan2(yt - X[1], xt - X[0]) - X[2])
    v = v_row if info["is_row"] and np.abs(info["preview"]).max() < 0.1 else v_turn
    return np.array([v, 2.0 * v * np.sin(alpha) / lookahead])


def run(P, robot, controller, X0=None, t_max=None, stop_at_end=True):
    """Closed loop: controller every DT_CTRL, robot integrated in between. Returns a log DataFrame."""
    X = P.start_state() if X0 is None else X0.copy()
    s_prev, rows = 0.0, []
    t_max = t_max if t_max is not None else 3.0 * P.length / 0.5
    t = 0.0
    while t < t_max:
        info = P.frame(X, s_prev)
        U = clip_command(controller(P, X, info))
        rows.append(dict(t=t, x=X[0], y=X[1], psi=X[2], v=X[3], w=X[4], v_c=U[0], w_c=U[1], s=info["s"],
                         e_y=info["e_y"], e_psi=info["e_psi"], d=info["d"], is_row=info["is_row"]))
        if stop_at_end and info["s"] >= P.length - 0.3:
            break
        X = robot.step(X, U)
        s_prev = info["s"]
        t += DT_CTRL
    return pd.DataFrame(rows)
