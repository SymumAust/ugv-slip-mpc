#!/usr/bin/env python3
"""
Feasibility check: does a learned slip residual improve 1-2 s motion prediction of the
CitrusFarm Jackal on a field it was not trained on?

Implements PLAN.md exactly (preprocessing, models, metric, splits, decision rules).
Usage:
    python feasibility.py --config config.json
    python feasibility.py --config config.json --quick      # fast smoke run (not for the decision)
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------------------
# Fixed constants from PLAN.md (do not change without logging an amendment in PLAN.md)
# ----------------------------------------------------------------------------------------
DT = 0.02                     # 50 Hz common grid
GPS_GAP_S = 0.5               # GPS gap that invalidates samples
STATIONARY_V, STATIONARY_W, STATIONARY_MIN_S = 0.01, 0.01, 1.0
STRAIGHT_V, STRAIGHT_W, STRAIGHT_GPS_SPEED = 0.5, 0.05, 0.3
HEADING_BIN_S = 60.0
MOVING_V, MOVING_W = 0.05, 0.05
WINDOW_STEP_S = 0.5
HORIZONS_S = (0.5, 1.0, 2.0)
TURN_W = 0.3
BOOT_BLOCK_S, BOOT_N, BOOT_SEED = 10.0, 1000, 0
GP_N_TRAIN, MLP_SEEDS = 2000, (0, 1, 2)
TRAIN_DOWNSAMPLE = 5          # 50 Hz -> 10 Hz for training samples
# decision thresholds
Q1_HEADING_RMS_DEG, Q2_MAX_LAG_S, Q3_MIN_MOVING_S = 3.0, 0.5, 300.0
S0_MIN_ERR_M = 0.15
G1_MIN_IMPR, G1_MIN_CI_LO = 0.30, 0.15
G2_MIN_IMPR, G2_MIN_CI_LO = 0.10, 0.0
G3_MIN_GAP = 0.15
G3_V_REF, G3_W_REF, G3_MIN_EFFECT = 1.2, 0.8, 0.05   # effect of a parameter change at typical driving

TOPIC_ODOM = "/jackal_velocity_controller/odom"
TOPIC_IMU = {"microstrain": ("base_*.bag", "/microstrain/imu/data"),
             "zed": ("zed_*.bag", "/zed2i/zed_node/imu/data")}
TOPIC_GT = "/gps/fix/odometry"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
def read_topic(paths, topic, extract):
    """Yield extract(msg, bag_time_s) for every message on `topic` in the given ROS1 bags."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path
    ts = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    with AnyReader([Path(p) for p in paths], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return rows
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            rows.append(extract(msg, t_ns * 1e-9))
    return rows


def stamp(msg, bag_t):
    s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return s if s > 0 else bag_t


def clean_time(t, name, flags):
    """Sort; repair duplicated stamps (known CitrusFarm IMU issue) by a uniform-rate fit."""
    order = np.argsort(t, kind="stable")
    t = t[order]
    dup = np.mean(np.diff(t) <= 0) if len(t) > 1 else 0.0
    if dup > 0.05:
        idx = np.arange(len(t))
        p = np.polyfit(idx, t, 1)
        t = np.polyval(p, idx)
        flags.append(f"{name}: {dup:.1%} non-increasing stamps, rebuilt with uniform-rate fit")
    return t, order


def load_odom(drive_dir, flags):
    paths = sorted(glob.glob(os.path.join(drive_dir, "odom_*.bag")))
    if not paths:
        raise FileNotFoundError(f"no odom_*.bag in {drive_dir}")
    rows = read_topic(paths, TOPIC_ODOM, lambda m, bt: (stamp(m, bt), m.twist.twist.linear.x,
                                                        m.twist.twist.linear.y, m.twist.twist.angular.z))
    a = np.array(rows, float)
    t, order = clean_time(a[:, 0], "odom", flags)
    a = a[order]
    if np.nanmax(np.abs(a[:, 2])) > 1e-3:
        flags.append("odom: nonzero twist.linear.y present (ignored)")
    return t, a[:, 1], a[:, 3]


def load_imu(drive_dir, prefer, flags):
    for src in prefer:
        pattern, topic = TOPIC_IMU[src]
        paths = sorted(glob.glob(os.path.join(drive_dir, pattern)))
        if not paths:
            continue
        rows = read_topic(paths, topic, lambda m, bt: (
            stamp(m, bt), m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
        if not rows:
            continue
        a = np.array(rows, float)
        t, order = clean_time(a[:, 0], f"imu[{src}]", flags)
        a = a[order]
        return t, a[:, 1:4], a[:, 4:7], src
    raise FileNotFoundError(f"no IMU found in {drive_dir} (looked for {prefer})")


def _gt_local(x, y, flags):
    if np.all(np.abs(x) <= 180) and np.all(np.abs(y) <= 180) and np.ptp(x) < 1 and np.ptp(y) < 1:
        # looks like lon/lat or lat/lon degrees -> local ENU (equirectangular)
        lat0 = np.deg2rad(np.mean(y)) if np.mean(np.abs(y)) < 90 else np.deg2rad(np.mean(x))
        flags.append("gt: coordinates looked like degrees; converted to local metres")
        R = 6378137.0
        return np.deg2rad(x - x[0]) * R * np.cos(lat0), np.deg2rad(y - y[0]) * R
    return x - x[0], y - y[0]


def load_gt(gt_dir, flags, local=True):
    bag = os.path.join(gt_dir, "gt.bag")
    csv = os.path.join(gt_dir, "gt.csv")
    if os.path.exists(bag):
        rows = read_topic([bag], TOPIC_GT, lambda m, bt: (stamp(m, bt), m.pose.pose.position.x,
                                                          m.pose.pose.position.y))
        a = np.array(rows, float)
        src = "gt.bag"
    elif os.path.exists(csv):
        df = pd.read_csv(csv, sep=None, engine="python", header=None, comment="#")
        if not np.issubdtype(df.iloc[0].dtype, np.number) or isinstance(df.iloc[0, 0], str):
            try:
                float(df.iloc[0, 0])
            except (TypeError, ValueError):
                header = [str(h).strip().lower() for h in df.iloc[0]]
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = header
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        cols = [str(c) for c in df.columns]

        def pick(names, default):
            for n in names:
                if n in cols:
                    return df[n].to_numpy(float)
            return df.iloc[:, default].to_numpy(float)
        t = pick(["timestamp", "time", "t", "#timestamp", "# timestamp"], 0)
        x = pick(["x", "position_x", "pos_x", "tx"], 1)
        y = pick(["y", "position_y", "pos_y", "ty"], 2)
        a = np.column_stack([t, x, y])
        src = "gt.csv"
    else:
        raise FileNotFoundError(f"no gt.bag or gt.csv in {gt_dir}")
    if np.median(a[:, 0]) > 1e17:      # nanoseconds
        a[:, 0] *= 1e-9
    t, order = clean_time(a[:, 0], "gt", flags)
    a = a[order]
    if not local:
        return t, a[:, 1], a[:, 2], src
    x, y = _gt_local(a[:, 1], a[:, 2], flags)
    return t, x, y, src


def load_processed(pdir, flags):
    """Load the compact extracts written by extract.py."""
    o = pd.read_csv(os.path.join(pdir, "odom.csv.gz"))
    i = pd.read_csv(os.path.join(pdir, "imu.csv.gz"))
    g = pd.read_csv(os.path.join(pdir, "gt.csv.gz"))
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    t_o, ord_o = clean_time(o.t.to_numpy(), "odom", flags)
    t_i, ord_i = clean_time(i.t.to_numpy(), "imu", flags)
    t_g, ord_g = clean_time(g.t.to_numpy(), "gt", flags)
    o, i, g = o.iloc[ord_o], i.iloc[ord_i], g.iloc[ord_g]
    x, y = _gt_local(g.x.to_numpy(), g.y.to_numpy(), flags)
    return ((t_o, o.v_w.to_numpy(), o.w_w.to_numpy()),
            (t_i, i[["gx", "gy", "gz"]].to_numpy(), i[["ax", "ay", "az"]].to_numpy(), meta["imu_source"]),
            (t_g, x, y, "processed:" + meta["gt_source"]))


# ----------------------------------------------------------------------------------------
# Preprocessing (PLAN.md section 3)
# ----------------------------------------------------------------------------------------
def xcorr_lag(t_ref, s_ref, t_oth, s_oth, max_lag=5.0):
    """Lag L maximising corr(s_ref(t), s_oth(t+L)). Corrected other times = t_oth - L."""
    lo, hi = max(t_ref[0], t_oth[0]) + max_lag, min(t_ref[-1], t_oth[-1]) - max_lag
    if hi - lo < 30:
        return 0.0, np.nan
    g = np.arange(lo, hi, DT)
    r = np.interp(g, t_ref, s_ref)
    r = (r - r.mean()) / (r.std() + 1e-12)

    def corr(L):
        o = np.interp(g + L, t_oth, s_oth)
        o = (o - o.mean()) / (o.std() + 1e-12)
        return np.mean(r * o)
    coarse = np.arange(-max_lag, max_lag + 1e-9, 0.1)
    c = [corr(L) for L in coarse]
    L0 = coarse[int(np.argmax(c))]
    fine = np.arange(L0 - 0.2, L0 + 0.2 + 1e-9, 0.01)
    cf = [corr(L) for L in fine]
    return float(fine[int(np.argmax(cf))]), float(np.max(cf))


def process_drive(name, field, cfg):
    flags = []
    root = cfg["data_root"]
    drive_dir = os.path.join(root, name)
    gt_dir = os.path.join(root, "ground_truth", name)
    lx, ly = cfg["lever_arm_xy"]

    pdir = os.path.join(cfg.get("processed_root", "data/processed"), name)
    if os.path.exists(os.path.join(pdir, "meta.json")):
        (t_o, v_o, w_o), (t_i, gyro, acc, imu_src), (t_g, x_g, y_g, gt_src) = load_processed(pdir, flags)
    else:
        t_o, v_o, w_o = load_odom(drive_dir, flags)
        t_i, gyro, acc, imu_src = load_imu(drive_dir, cfg.get("imu_preference", ["microstrain", "zed"]), flags)
        t_g, x_g, y_g, gt_src = load_gt(gt_dir, flags)
    log(f"{name}: odom {len(t_o)}, imu[{imu_src}] {len(t_i)}, gt[{gt_src}] {len(t_g)} msgs")

    # 1. IMU up axis from mean specific force
    up = acc.mean(axis=0)
    g_norm = np.linalg.norm(up)
    up = up / g_norm
    if abs(g_norm - 9.81) > 1.0:
        flags.append(f"imu: mean |acc| = {g_norm:.2f} m/s^2 (expected ~9.81)")
    yaw_raw = gyro @ up

    # 3a. IMU lag vs odom (on yaw rate)
    lag_imu, c_imu = xcorr_lag(t_o, w_o, t_i, yaw_raw)
    if abs(lag_imu) > 0.02:
        t_i = t_i - lag_imu
    # 2. gyro bias from stationary periods
    from scipy.ndimage import median_filter
    v_med, w_med = median_filter(np.abs(v_o), 25), median_filter(np.abs(w_o), 25)   # 0.5 s moving median
    v_at_i = np.interp(t_i, t_o, v_med)
    w_at_i = np.interp(t_i, t_o, w_med)
    still = (v_at_i < STATIONARY_V) & (w_at_i < STATIONARY_W)
    still_long = np.zeros_like(still)
    if still.any():
        edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if t_i[e - 1] - t_i[s] >= STATIONARY_MIN_S:
                still_long[s:e] = True
    if still_long.sum() > 50:
        bias = float(np.median(yaw_raw[still_long]))
    else:
        bias = 0.0
        flags.append("imu: no stationary period >= 1 s; gyro bias set to 0")
    yaw_rate_i = yaw_raw - bias

    # GPS on a uniform 10 Hz grid (for derivatives) with gap tracking
    gaps = np.flatnonzero(np.diff(t_g) > GPS_GAP_S)
    t10 = np.arange(t_g[0], t_g[-1], 0.1)
    x10, y10 = np.interp(t10, t_g, x_g), np.interp(t10, t_g, y_g)
    vx10 = savgol_filter(x10, 7, 2, deriv=1, delta=0.1)
    vy10 = savgol_filter(y10, 7, 2, deriv=1, delta=0.1)
    sp10 = np.hypot(vx10, vy10)

    # 3b. GPS lag vs odom (on speed)
    lag_gps, c_gps = xcorr_lag(t_o, np.abs(v_o), t10, sp10)
    if abs(lag_gps) > 0.02:
        t_g, t10 = t_g - lag_gps, t10 - lag_gps

    # 4. common 50 Hz grid
    t0 = max(t_o[0], t_i[0], t_g[0]) + 1.0
    t1 = min(t_o[-1], t_i[-1], t_g[-1]) - 1.0
    t = np.arange(t0, t1, DT)
    valid = np.ones_like(t, bool)
    for k in gaps:
        valid &= ~((t > t_g[k] - 0.2) & (t < t_g[k + 1] + 0.2))
    v_w, w_w = np.interp(t, t_o, v_o), np.interp(t, t_o, w_o)
    w_true = np.interp(t, t_i, yaw_rate_i)
    px, py = np.interp(t, t_g, x_g), np.interp(t, t_g, y_g)          # raw (for errors)
    vxw, vyw = np.interp(t, t10, vx10), np.interp(t, t10, vy10)       # smoothed velocity
    gps_speed = np.hypot(vxw, vyw)
    # feature derivatives (smoothed, 0.22 s window)
    dv_w = savgol_filter(v_w, 11, 2, deriv=1, delta=DT)
    dw_w = savgol_filter(w_w, 11, 2, deriv=1, delta=DT)

    # 6. heading: integrated gyro + slowly varying offset fitted to GPS course
    psi_imu = np.concatenate([[0.0], np.cumsum(0.5 * (w_true[1:] + w_true[:-1]) * DT)])
    course = np.arctan2(vyw, vxw)
    fwd = valid & (v_w > STRAIGHT_V) & (np.abs(w_w) < STRAIGHT_W) & (gps_speed > STRAIGHT_GPS_SPEED)
    rev = valid & (v_w < -STRAIGHT_V) & (np.abs(w_w) < STRAIGHT_W) & (gps_speed > STRAIGHT_GPS_SPEED)
    straight = fwd | rev
    d = wrap(course - psi_imu + np.where(rev, np.pi, 0.0))
    heading_rms = np.nan
    if straight.sum() > 100:
        ds = np.unwrap(d[straight])
        ts_ = t[straight]
        bins = np.floor((ts_ - t[0]) / HEADING_BIN_S).astype(int)
        cen, med = [], []
        for b in np.unique(bins):
            m = bins == b
            if m.sum() >= 50:
                cen.append(np.median(ts_[m]))
                med.append(np.median(ds[m]))
        if cen:
            med = np.unwrap(np.array(med))
            offset = np.interp(t, cen, med)
            psi = psi_imu + offset
            res = wrap(course - psi + np.where(rev, np.pi, 0.0))[straight]
            heading_rms = float(np.degrees(np.sqrt(np.mean(res ** 2))))
        else:
            psi = psi_imu
    else:
        psi = psi_imu
        flags.append("heading: too few straight moving samples to align heading")

    # 7. true body velocity at base_link
    c, s = np.cos(psi), np.sin(psi)
    vax, vay = c * vxw + s * vyw, -s * vxw + c * vyw
    vx_true = vax + w_true * ly
    vy_true = vay - w_true * lx

    moving = valid & ((np.abs(v_w) >= MOVING_V) | (np.abs(w_w) >= MOVING_W))
    D = dict(name=name, field=field, t=t, valid=valid, moving=moving,
             F=np.column_stack([v_w, w_w, dv_w, dw_w]),
             Y=np.column_stack([vx_true, vy_true, w_true]),
             px=px, py=py, psi=psi)
    Q = dict(drive=name, field=field, imu_source=imu_src, gt_source=gt_src,
             duration_s=float(t[-1] - t[0]), moving_s=float(moving.sum() * DT),
             lag_gps_s=lag_gps, corr_gps=c_gps, lag_imu_s=lag_imu, corr_imu=c_imu,
             gyro_bias_rad_s=bias, heading_rms_deg=heading_rms, n_gps_gaps=int(len(gaps)),
             flags=" | ".join(flags))
    Q["Q1"] = bool(np.isfinite(heading_rms) and heading_rms < Q1_HEADING_RMS_DEG)
    Q["Q2"] = bool(abs(lag_gps) < Q2_MAX_LAG_S and abs(lag_imu) < Q2_MAX_LAG_S)
    Q["Q3"] = bool(Q["moving_s"] >= Q3_MIN_MOVING_S)
    Q["passed"] = Q["Q1"] and Q["Q2"] and Q["Q3"]
    return D, Q


# ----------------------------------------------------------------------------------------
# Models (PLAN.md section 4)
# ----------------------------------------------------------------------------------------
class Nominal:
    name = "M0-nominal"

    def fit(self, F, Y):
        return self

    def predict(self, F):
        return np.column_stack([F[:, 0], np.zeros(len(F)), F[:, 1]])


class ICR:
    name = "M1-ICR"

    def fit(self, F, Y):
        v, w = F[:, 0], F[:, 1]
        self.a = float(v @ Y[:, 0] / (v @ v))
        self.c = float(w @ Y[:, 1] / (w @ w))
        self.b = float(w @ Y[:, 2] / (w @ w))
        return self

    def predict(self, F):
        return np.column_stack([self.a * F[:, 0], self.c * F[:, 1], self.b * F[:, 1]])


class Residual:
    def __init__(self, kind, n_gp=GP_N_TRAIN):
        self.kind, self.n_gp = kind, n_gp
        self.name = "M2-GP" if kind == "gp" else "M2-MLP"

    def fit(self, F, Y):
        self.base = ICR().fit(F, Y)
        R = Y - self.base.predict(F)
        self.xm, self.xs = F.mean(0), F.std(0) + 1e-9
        self.rm, self.rs = R.mean(0), R.std(0) + 1e-9
        X, Rn = (F - self.xm) / self.xs, (R - self.rm) / self.rs
        if self.kind == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), size=min(self.n_gp, len(X)), replace=False)
            self.models = []
            for j in range(3):
                k = C(1.0, (1e-3, 1e2)) * RBF(np.ones(X.shape[1]), (1e-2, 1e2)) + WhiteKernel(0.1, (1e-5, 1e1))
                gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=1, random_state=0)
                gp.fit(X[idx], Rn[idx, j])
                self.models.append(gp)
        else:
            from sklearn.neural_network import MLPRegressor
            self.models = [MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=500, random_state=s).fit(X, Rn)
                           for s in MLP_SEEDS]
        return self

    def predict(self, F):
        X = (F - self.xm) / self.xs
        out = np.empty((len(F), 3))
        for s in range(0, len(F), 5000):
            sl = slice(s, s + 5000)
            if self.kind == "gp":
                out[sl] = np.column_stack([m.predict(X[sl]) for m in self.models])
            else:
                out[sl] = np.mean([m.predict(X[sl]) for m in self.models], axis=0)
        return self.base.predict(F) + out * self.rs + self.rm


def training_set(drives):
    F = np.vstack([D["F"][D["moving"]][::TRAIN_DOWNSAMPLE] for D in drives])
    Y = np.vstack([D["Y"][D["moving"]][::TRAIN_DOWNSAMPLE] for D in drives])
    return F, Y


# ----------------------------------------------------------------------------------------
# Evaluation (PLAN.md section 5)
# ----------------------------------------------------------------------------------------
def windows(D):
    step, H = int(round(WINDOW_STEP_S / DT)), int(round(max(HORIZONS_S) / DT))
    ok = D["valid"] & D["moving"]
    cs = np.concatenate([[0], np.cumsum(~D["valid"])])
    starts = []
    for i in range(0, len(D["t"]) - H - 1, step):
        if ok[i] and cs[i + H + 1] - cs[i] == 0:
            starts.append(i)
    return np.array(starts, int)


def rollout(D, Vb, starts, lever):
    lx, ly = lever
    H = int(round(max(HORIZONS_S) / DT))
    hs = {h: int(round(h / DT)) for h in HORIZONS_S}
    psi = D["psi"][starts].copy()
    bx = D["px"][starts] - (np.cos(psi) * lx - np.sin(psi) * ly)
    by = D["py"][starts] - (np.sin(psi) * lx + np.cos(psi) * ly)
    pos_err, head_err = {}, {}
    for k in range(H + 1):
        if k in hs.values():
            h = [hh for hh, kk in hs.items() if kk == k][0]
            ax = bx + np.cos(psi) * lx - np.sin(psi) * ly
            ay = by + np.sin(psi) * lx + np.cos(psi) * ly
            pos_err[h] = np.hypot(ax - D["px"][starts + k], ay - D["py"][starts + k])
            head_err[h] = np.abs(wrap(psi - D["psi"][starts + k]))
        if k == H:
            break
        v = Vb[starts + k]
        c, s = np.cos(psi), np.sin(psi)
        bx = bx + (c * v[:, 0] - s * v[:, 1]) * DT
        by = by + (s * v[:, 0] + c * v[:, 1]) * DT
        psi = psi + v[:, 2] * DT
    turn = np.array([np.max(np.abs(D["F"][i:i + H + 1, 1])) > TURN_W for i in starts])
    return pos_err, head_err, turn


def improvement_ci(eA, eB, blocks, n_boot=BOOT_N):
    point = 1 - eA.mean() / eB.mean()
    rng = np.random.default_rng(BOOT_SEED)
    ub = np.unique(blocks)
    idx_by = {b: np.flatnonzero(blocks == b) for b in ub}
    sA = np.array([eA[idx_by[b]].sum() for b in ub])
    sB = np.array([eB[idx_by[b]].sum() for b in ub])
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        reps.append(1 - sA[pick].sum() / sB[pick].sum())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def icr_bootstrap(drives, n_boot=BOOT_N):
    F, Y = training_set(drives)
    tt = np.concatenate([D["t"][D["moving"]][::TRAIN_DOWNSAMPLE] + 1e7 * i for i, D in enumerate(drives)])
    blocks = np.floor(tt / BOOT_BLOCK_S).astype(np.int64)
    ub = np.unique(blocks)
    idx_by = [np.flatnonzero(blocks == b) for b in ub]
    m = ICR().fit(F, Y)
    rng = np.random.default_rng(BOOT_SEED)
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[j] for j in rng.integers(0, len(ub), len(ub))])
        mm = ICR().fit(F[idx], Y[idx])
        reps.append([mm.a, mm.b, mm.c])
    reps = np.array(reps)
    lo, hi = np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0)
    return {"a": (m.a, lo[0], hi[0]), "b": (m.b, lo[1], hi[1]), "c": (m.c, lo[2], hi[2])}


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--quick", action="store_true", help="smoke run: small GP, 200 bootstrap reps")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = cfg.get("output_dir", "results")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    n_boot = 200 if args.quick else BOOT_N
    n_gp = 500 if args.quick else GP_N_TRAIN

    provisional = []
    if cfg.get("lever_arm_xy") is None:
        if not cfg.get("allow_zero_lever_arm", False):
            sys.exit("lever_arm_xy missing in config. Fill it from the CitrusFarm calibration files "
                     "(see README), or set allow_zero_lever_arm=true for a PROVISIONAL run.")
        cfg["lever_arm_xy"] = [0.0, 0.0]
        provisional.append("lever arm assumed (0, 0): lateral-slip (c) and v_y results are not trustworthy")
    if args.quick:
        provisional.append("--quick run: reduced GP size and bootstrap; not valid for the decision")
    lever = tuple(cfg["lever_arm_xy"])

    # ---- load & preprocess
    drives, qrows = [], []
    for name, field in cfg["drives"].items():
        if not (os.path.isdir(os.path.join(cfg["data_root"], name)) or
                os.path.exists(os.path.join(cfg.get("processed_root", "data/processed"), name, "meta.json"))):
            log(f"{name}: folder not found, skipped")
            continue
        try:
            D, Q = process_drive(name, field, cfg)
        except Exception as e:  # report and continue with the other drives
            log(f"{name}: FAILED to load/process: {e}")
            qrows.append(dict(drive=name, field=field, passed=False, flags=f"error: {e}"))
            continue
        qrows.append(Q)
        log(f"{name}: heading RMS {Q['heading_rms_deg']:.2f} deg, lag gps {Q['lag_gps_s']:+.2f} s, "
            f"lag imu {Q['lag_imu_s']:+.2f} s, moving {Q['moving_s']:.0f} s -> "
            f"{'PASS' if Q['passed'] else 'FAIL'}")
        if Q["passed"]:
            drives.append(D)
    pd.DataFrame(qrows).to_csv(os.path.join(out, "data_quality.csv"), index=False)

    fields = sorted({D["field"] for D in drives})
    by_field = {f: [D for D in drives if D["field"] == f] for f in fields}
    summary = dict(provisional=provisional, lever_arm_xy=lever, fields={f: [D["name"] for D in by_field[f]]
                                                                          for f in fields})
    if len(fields) < 2:
        verdict = "NO-GO"
        reasons = [f"fewer than two fields with a usable drive after quality gate: {summary['fields']}"]
        write_decision(out, verdict, reasons, {}, summary, qrows)
        return

    # ---- ICR params per field (for G3b and the report)
    icr_rows, icr_by_field = [], {}
    for f in fields:
        p = icr_bootstrap(by_field[f], n_boot)
        icr_by_field[f] = p
        for k, (v, lo, hi) in p.items():
            icr_rows.append(dict(field=f, param=k, value=v, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(icr_rows).to_csv(os.path.join(out, "icr_params.csv"), index=False)

    def make_models():
        return [Nominal(), ICR(), Residual("gp", n_gp), Residual("mlp")]

    # ---- cross-field evaluation
    err_rows, pooled = [], {}
    rollouts_cache = {}
    for f_tr in fields:
        for f_te in fields:
            if f_tr == f_te:
                continue
            direction = f"{f_tr}->{f_te}"
            F, Y = training_set(by_field[f_tr])
            log(f"[{direction}] training on {len(F)} samples from {[D['name'] for D in by_field[f_tr]]}")
            models = [m.fit(F, Y) for m in make_models()]
            for D in by_field[f_te]:
                st = windows(D)
                for m in models:
                    Vb = m.predict(D["F"])
                    pe, he, turn = rollout(D, Vb, st, lever)
                    blocks = (np.floor(D["t"][st] / BOOT_BLOCK_S).astype(np.int64)
                              + (zlib.crc32(D["name"].encode()) % 10**6) * 10**9)
                    pooled.setdefault((direction, m.name), []).append((pe, he, turn, blocks))
                    rollouts_cache[(direction, D["name"], m.name)] = (st, Vb)
                    for h in HORIZONS_S:
                        for lab, msk in (("all", np.ones_like(turn)), ("straight", ~turn), ("turn", turn)):
                            if msk.sum() == 0:
                                continue
                            err_rows.append(dict(split="cross-field", direction=direction, test_drive=D["name"],
                                                 model=m.name, horizon_s=h, subset=lab, n_windows=int(msk.sum()),
                                                 pos_err_mean_m=float(pe[h][msk].mean()),
                                                 pos_err_median_m=float(np.median(pe[h][msk])),
                                                 head_err_mean_deg=float(np.degrees(he[h][msk].mean()))))

    def pool(direction, mname, h):
        parts = pooled[(direction, mname)]
        return (np.concatenate([p[0][h] for p in parts]), np.concatenate([p[3] for p in parts]))

    directions = sorted({k[0] for k in pooled})
    stats = {}
    for dname in directions:
        e0, blocks = pool(dname, "M0-nominal", 2.0)
        e1, _ = pool(dname, "M1-ICR", 2.0)
        eg, _ = pool(dname, "M2-GP", 2.0)
        em, _ = pool(dname, "M2-MLP", 2.0)
        stats[dname] = dict(
            n_windows=int(len(e0)), err_M0=float(e0.mean()), err_M1=float(e1.mean()),
            err_GP=float(eg.mean()), err_MLP=float(em.mean()),
            GP_vs_M0=improvement_ci(eg, e0, blocks, n_boot), GP_vs_M1=improvement_ci(eg, e1, blocks, n_boot),
            MLP_vs_M0=improvement_ci(em, e0, blocks, n_boot), MLP_vs_M1=improvement_ci(em, e1, blocks, n_boot),
            M1_vs_M0=improvement_ci(e1, e0, blocks, n_boot))

    # ---- within-field evaluation (leave-one-drive-out inside a field): G1, G2, G3a
    gaps, wpool = [], {}
    for f in fields:
        if len(by_field[f]) < 2:
            continue
        other = [g for g in fields if g != f][0]
        for D in by_field[f]:
            rest = [E for E in by_field[f] if E["name"] != D["name"]]
            F, Y = training_set(rest)
            st = windows(D)
            blocks = (np.floor(D["t"][st] / BOOT_BLOCK_S).astype(np.int64)
                      + (zlib.crc32(D["name"].encode()) % 10**6) * 10**9)
            for m in make_models():
                m.fit(F, Y)
                pe, he, turn = rollout(D, m.predict(D["F"]), st, lever)
                wpool.setdefault(m.name, []).append((pe, he, turn, blocks))
                for h in HORIZONS_S:
                    for lab, msk in (("all", np.ones_like(turn)), ("straight", ~turn), ("turn", turn)):
                        if msk.sum() == 0:
                            continue
                        err_rows.append(dict(split="within-field", direction=f"{f}->{f}", test_drive=D["name"],
                                             model=m.name, horizon_s=h, subset=lab, n_windows=int(msk.sum()),
                                             pos_err_mean_m=float(pe[h][msk].mean()),
                                             pos_err_median_m=float(np.median(pe[h][msk])),
                                             head_err_mean_deg=float(np.degrees(he[h][msk].mean()))))
                if m.name == "M2-GP":
                    e_in = pe[2.0]
            _, Vb_cross = rollouts_cache[(f"{other}->{f}", D["name"], "M2-GP")]
            e_cross = rollout(D, Vb_cross, st, lever)[0][2.0]
            gaps.append(dict(test_drive=D["name"], err_within=float(e_in.mean()),
                             err_cross=float(e_cross.mean()), gap=float(e_cross.mean() / e_in.mean() - 1)))
    within = None
    if wpool:
        cat = lambda name, i, h=2.0: np.concatenate([p[i][h] if i < 2 else p[i] for p in wpool[name]])
        e0, e1, eg, em = (cat(n, 0) for n in ("M0-nominal", "M1-ICR", "M2-GP", "M2-MLP"))
        blocks = np.concatenate([p[3] for p in wpool["M0-nominal"]])
        turn = np.concatenate([p[2] for p in wpool["M0-nominal"]])
        within = dict(n_windows=int(len(e0)), n_test_drives=len(wpool["M0-nominal"]),
                      err_M0=float(e0.mean()), err_M1=float(e1.mean()), err_GP=float(eg.mean()),
                      err_MLP=float(em.mean()),
                      GP_vs_M0=improvement_ci(eg, e0, blocks, n_boot), GP_vs_M1=improvement_ci(eg, e1, blocks, n_boot),
                      MLP_vs_M1=improvement_ci(em, e1, blocks, n_boot), M1_vs_M0=improvement_ci(e1, e0, blocks, n_boot))
    pd.DataFrame(err_rows).to_csv(os.path.join(out, "errors_by_model.csv"), index=False)

    # ---- decision rules (PLAN.md section 7, as amended 2026-11-13)
    R = {}
    turn_err_M0 = {}
    for d in directions:
        parts = pooled[(d, "M0-nominal")]
        e = np.concatenate([p[0][2.0] for p in parts])
        tm = np.concatenate([p[2] for p in parts])
        turn_err_M0[d] = float(e[tm].mean()) if tm.any() else float("nan")
    R["S0"] = all(np.isfinite(v) and v >= S0_MIN_ERR_M for v in turn_err_M0.values())
    if within is None or within["n_test_drives"] < 2:
        R["G1"] = R["G2"] = False
        reasons_extra = ["within-field test needs >= 2 drives in at least one field; G1/G2 cannot be evaluated"]
    else:
        reasons_extra = []
        R["G1"] = bool(within["GP_vs_M0"][0] >= G1_MIN_IMPR and within["GP_vs_M0"][1] >= G1_MIN_CI_LO)
        R["G2"] = bool(within["GP_vs_M1"][0] >= G2_MIN_IMPR and within["GP_vs_M1"][1] > G2_MIN_CI_LO)
    R["G4"] = all(stats[d]["GP_vs_M1"][0] >= G2_MIN_IMPR and stats[d]["GP_vs_M1"][1] > G2_MIN_CI_LO
                  for d in directions)
    g3a = float(np.mean([g["gap"] for g in gaps])) if gaps else np.nan
    G3a = bool(np.isfinite(g3a) and g3a >= G3_MIN_GAP)
    G3b_detail = {}
    f1, f2 = fields[0], fields[1]
    for k in ("a", "b", "c"):
        v1, lo1, hi1 = icr_by_field[f1][k]
        v2, lo2, hi2 = icr_by_field[f2][k]
        effect = abs(v1 - v2) * (G3_V_REF if k == "a" else G3_W_REF)   # m/s for a, c; rad/s for b
        sep = hi1 < lo2 or hi2 < lo1
        G3b_detail[k] = dict(effect=effect, ci_separated=sep, ok=bool(effect >= G3_MIN_EFFECT and sep))
    G3b = any(v["ok"] for v in G3b_detail.values())
    R["G3"] = bool(G3a or G3b)

    if not R["S0"]:
        verdict = "NO-GO"
    elif R["G1"] and R["G2"] and R["G3"]:
        verdict = "GO-FULL"
    elif R["G1"] and R["G2"]:
        verdict = "GO-OFFLINE"
    elif not R["G2"] and R["G3"]:
        verdict = "PIVOT-ADAPTIVE"
    else:  # G2 and G3 both fail, or G2 passes but G1 fails (gain over nominal < 30 %)
        verdict = "NO-GO"
    summary.update(directions=stats, within=within, turn_err_M0_2s=turn_err_M0, within_field_gaps=gaps,
                   G3a_mean_gap=g3a, G3b=G3b_detail, rules=R, icr_params={f: icr_by_field[f] for f in fields})
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    write_decision(out, verdict, reasons_extra, R, summary, qrows)
    make_figures(out, drives, by_field, fields, icr_by_field, err_rows, rollouts_cache, lever)
    log(f"VERDICT: {verdict}  (see {out}/decision.md)")


def write_decision(out, verdict, reasons, R, summary, qrows):
    L = [f"# Feasibility decision: **{verdict}**", "",
         "Rules applied exactly as pre-registered in PLAN.md. Decision uses M2-GP only.", ""]
    if summary.get("provisional"):
        L += ["> **PROVISIONAL RESULT, NOT VALID FOR THE DECISION:**"] + [f"> - {p}" for p in summary["provisional"]] + [""]
    L += ["## Data-quality gate", "", "| Drive | Field | Heading RMS (deg) | GPS lag (s) | IMU lag (s) | Moving (s) | Result |",
          "|---|---|---|---|---|---|---|"]
    for q in qrows:
        L.append(f"| {q.get('drive')} | {q.get('field')} | {q.get('heading_rms_deg', float('nan')):.2f} | "
                 f"{q.get('lag_gps_s', float('nan')):+.2f} | {q.get('lag_imu_s', float('nan')):+.2f} | "
                 f"{q.get('moving_s', float('nan')):.0f} | {'PASS' if q.get('passed') else 'FAIL'} |")
    flagged = [q for q in qrows if q.get("flags")]
    if flagged:
        L += ["", "Flags:"] + [f"- {q['drive']}: {q['flags']}" for q in flagged]
    L += [""] + [f"- {r}" for r in reasons]
    if R:
        f = lambda x: f"{x[0]:+.1%} [{x[1]:+.1%}, {x[2]:+.1%}]"
        w = summary.get("within")
        if w:
            L += ["", "## Within-field results at H = 2 s (leave-one-drive-out; basis for G1, G2)", "",
                  "| Test drives | Windows | M0 (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |",
                  "|---|---|---|---|---|---|---|---|---|",
                  f"| {w['n_test_drives']} | {w['n_windows']} | {w['err_M0']:.3f} | {w['err_M1']:.3f} | {w['err_GP']:.3f} | "
                  f"{w['err_MLP']:.3f} | {f(w['GP_vs_M0'])} | {f(w['GP_vs_M1'])} | {w['MLP_vs_M1'][0]:+.1%} |"]
        L += ["", "## Cross-field results at H = 2 s (train on one field, test on the other; basis for G4)", "",
              "| Direction | Windows | M0 (m) | M0 turning (m) | M1 (m) | GP (m) | MLP (m) | GP vs M0 [95% CI] | GP vs M1 [95% CI] | MLP vs M1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for d, s in summary["directions"].items():
            L.append(f"| {d} | {s['n_windows']} | {s['err_M0']:.3f} | {summary['turn_err_M0_2s'][d]:.3f} | "
                     f"{s['err_M1']:.3f} | {s['err_GP']:.3f} | "
                     f"{s['err_MLP']:.3f} | {f(s['GP_vs_M0'])} | {f(s['GP_vs_M1'])} | {s['MLP_vs_M1'][0]:+.1%} |")
        L += ["", "## Rules", "",
              f"- **S0** (M0 error on turning windows ≥ {S0_MIN_ERR_M} m, every cross-field direction): {'PASS' if R['S0'] else 'FAIL'}",
              f"- **G1** (within-field GP vs M0 ≥ {G1_MIN_IMPR:.0%}, CI low ≥ {G1_MIN_CI_LO:.0%}): {'PASS' if R['G1'] else 'FAIL'}",
              f"- **G2** (within-field GP vs M1 ≥ {G2_MIN_IMPR:.0%}, CI low > 0): {'PASS' if R['G2'] else 'FAIL'}",
              f"- **G4** (cross-field GP vs M1 ≥ {G2_MIN_IMPR:.0%}, CI low > 0, both directions; reported, not a gate): "
              f"{'PASS: offline model already generalises' if R['G4'] else 'FAIL: offline model does not transfer across fields'}",
              f"- **G3** (adaptation signal): {'PASS' if R['G3'] else 'FAIL'}",
              f"  - G3a mean cross-vs-within gap = {summary['G3a_mean_gap']:+.1%} (needs ≥ {G3_MIN_GAP:.0%}); "
              f"per drive: " + ", ".join(f"{g['test_drive']} {g['gap']:+.1%}" for g in summary["within_field_gaps"]),
              f"  - G3b ICR parameter effect between fields (needs ≥ {G3_MIN_EFFECT} m/s or rad/s at v={G3_V_REF}, ω={G3_W_REF}): " + ", ".join(
                  f"{k}: effect {v['effect']:.3f}{' (CIs separated)' if v['ci_separated'] else ''}{' OK' if v['ok'] else ''}"
                  for k, v in summary["G3b"].items()),
              "", "## Calibrated ICR parameters (v_x = a·v_w, ω = b·ω_w, v_y = c·ω_w)", "",
              "| Field | a | b | c |", "|---|---|---|---|"]
        for fld, p in summary["icr_params"].items():
            L.append(f"| {fld} | " + " | ".join(f"{p[k][0]:.3f} [{p[k][1]:.3f}, {p[k][2]:.3f}]" for k in "abc") + " |")
    nxt = {"GO-FULL": "Proceed: learned-residual MPC with online updating vs offline-only; cross-field generalisation.",
           "GO-OFFLINE": "Proceed: learned-residual MPC; online updating only as an ablation, not a claim.",
           "PIVOT-ADAPTIVE": "Pivot: online adaptive ICR-parameter MPC (physics model, parameters learned online).",
           "NO-GO": "Switch to idea #2: neural approximation of MPC with a safety projection."}
    L += ["", f"## Next step", "", nxt[verdict]]
    with open(os.path.join(out, "decision.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def make_figures(out, drives, by_field, fields, icr_by_field, err_rows, cache, lever):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = os.path.join(out, "figures")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1. slip vs commanded turn rate
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, D in enumerate(drives):
        m = D["moving"]
        sel = np.flatnonzero(m)[::10]
        F, Y = D["F"][sel], D["Y"][sel]
        kw = dict(s=3, alpha=0.3, color=colors[i % len(colors)], label=f"{D['name']}")
        ax[0].scatter(F[:, 1], Y[:, 2] - F[:, 1], **kw)
        ax[1].scatter(F[:, 1], Y[:, 1], **kw)
        ax[2].scatter(F[:, 1], Y[:, 0] - F[:, 0], **kw)
    ww = np.linspace(-1.5, 1.5, 50)
    for j, f in enumerate(fields):
        p = icr_by_field[f]
        ax[0].plot(ww, (p["b"][0] - 1) * ww, "k-" if j == 0 else "k--", lw=1.5, label=f"M1 fit {f}")
        ax[1].plot(ww, p["c"][0] * ww, "k-" if j == 0 else "k--", lw=1.5)
    ax[0].set(xlabel="wheel ω_w (rad/s)", ylabel="Δω = ω_true − ω_w (rad/s)", title="Yaw-rate slip")
    ax[1].set(xlabel="wheel ω_w (rad/s)", ylabel="v_y at base_link (m/s)", title="Lateral velocity")
    ax[2].set(xlabel="wheel ω_w (rad/s)", ylabel="Δv_x = v_x − v_w (m/s)", title="Longitudinal slip")
    ax[0].legend(markerscale=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "slip_vs_omega.png"), dpi=130)
    plt.close(fig)

    # 2. per-drive slip distribution (turning: Δω/ω_w ; straight: Δv_x/v_w)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    turn_r, str_r, labels = [], [], []
    for D in drives:
        F, Y, m = D["F"], D["Y"], D["moving"]
        t = m & (np.abs(F[:, 1]) > TURN_W)
        s = m & (np.abs(F[:, 1]) < STRAIGHT_W) & (np.abs(F[:, 0]) > STRAIGHT_V)
        turn_r.append((Y[t, 2] / F[t, 1]) if t.any() else [np.nan])
        str_r.append((Y[s, 0] / F[s, 0]) if s.any() else [np.nan])
        labels.append(f"{D['name'][:6]}")
    ax[0].boxplot(turn_r, showfliers=False)
    ax[0].set_xticks(range(1, len(labels) + 1), labels)
    ax[0].set(title="Turning: ω_true / ω_w", ylabel="ratio (1 = no slip)")
    ax[1].boxplot(str_r, showfliers=False)
    ax[1].set_xticks(range(1, len(labels) + 1), labels)
    ax[1].set(title="Straight: v_x / v_w", ylabel="ratio (1 = no slip)")
    for a in ax:
        a.axhline(1, color="grey", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "slip_per_drive.png"), dpi=130)
    plt.close(fig)

    # 3. error vs horizon per model, per direction
    df = pd.DataFrame(err_rows)
    df = df[(df.split == "cross-field") & (df.subset == "all")]
    dirs = sorted(df.direction.unique())
    fig, ax = plt.subplots(1, len(dirs), figsize=(5.5 * len(dirs), 4), squeeze=False)
    for j, d in enumerate(dirs):
        sub = df[df.direction == d]
        for k, (mname, g) in enumerate(sub.groupby("model")):
            w = g.groupby("horizon_s").apply(lambda x: np.average(x.pos_err_mean_m, weights=x.n_windows))
            ax[0, j].plot(w.index, w.values, "o-", color=colors[k], label=mname)
        ax[0, j].set(title=f"Train {d.split('->')[0]} → test {d.split('->')[1]}", xlabel="horizon (s)",
                     ylabel="mean position error (m)")
        ax[0, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "error_vs_horizon.png"), dpi=130)
    plt.close(fig)

    # 4. example U-turn rollout on the first cross-field test drive
    key = sorted({(k[0], k[1]) for k in cache})[0]
    D = [E for E in drives if E["name"] == key[1]][0]
    st = cache[(key[0], key[1], "M0-nominal")][0]
    turniness = [np.sum(np.abs(D["F"][i:i + 100, 1])) for i in st]
    i0 = st[int(np.argmax(turniness))]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sl = slice(i0, i0 + 101)
    ax.plot(D["px"][sl], D["py"][sl], "k-", lw=2.5, label="RTK GPS (truth)")
    for k, mname in enumerate(["M0-nominal", "M1-ICR", "M2-GP", "M2-MLP"]):
        _, Vb = cache[(key[0], key[1], mname)]
        lx, ly = lever
        psi = D["psi"][i0]
        bx = D["px"][i0] - (np.cos(psi) * lx - np.sin(psi) * ly)
        by = D["py"][i0] - (np.sin(psi) * lx + np.cos(psi) * ly)
        xs, ys = [], []
        for j in range(101):
            xs.append(bx + np.cos(psi) * lx - np.sin(psi) * ly)
            ys.append(by + np.sin(psi) * lx + np.cos(psi) * ly)
            v = Vb[i0 + j]
            bx += (np.cos(psi) * v[0] - np.sin(psi) * v[1]) * DT
            by += (np.sin(psi) * v[0] + np.cos(psi) * v[1]) * DT
            psi += v[2] * DT
        ax.plot(xs, ys, "--", color=colors[k], label=mname)
    ax.set(aspect="equal", title=f"2 s open-loop rollout, sharpest turn ({D['name']})", xlabel="x (m)", ylabel="y (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "example_rollout.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
