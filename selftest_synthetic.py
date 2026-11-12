#!/usr/bin/env python3
"""
Self-test: write synthetic drives in the exact CitrusFarm bag format (same topics, message types,
folder layout) with a KNOWN slip model, then run feasibility.py on them.

What the synthetic data deliberately contains, to test each preprocessing step:
  - nonlinear, field-dependent skid-steer slip (so M1 is imperfect and fields differ)
  - IMU mounted upside down and rotated (tests the up-axis detection), with gyro bias
  - GPS antenna lever arm and a +0.08 s GPS timestamp lag (tests lag estimation)
  - RTK noise 1 cm, odometry and IMU noise, a short stationary period at the start

Usage:
    python selftest_synthetic.py --out synth_slip   --slip yes
    python selftest_synthetic.py --out synth_strong --slip strong
    python selftest_synthetic.py --out synth_noslip --slip no
then  python feasibility.py --config synth_slip/config.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)
T = TS.types
LEVER = (-0.12, 0.05)          # true GPS antenna position in base_link (m)
GPS_LAG = 0.08                 # recorded GPS stamps are late by this much (s)
FIELD_SLIP = {                 # true slip parameters per field
    "13B": dict(b=0.80, k=0.10, c=0.06, a0=0.97, aw=0.05),
    "13D": dict(b=0.68, k=0.16, c=0.09, a0=0.95, aw=0.08),
}
NO_SLIP = dict(b=1.0, k=0.0, c=0.0, a0=1.0, aw=0.0)
STRONG_SLIP = {
    "13B": dict(b=0.65, k=0.12, c=0.08, a0=0.96, aw=0.08),
    "13D": dict(b=0.50, k=0.20, c=0.12, a0=0.93, aw=0.12),
}


def hdr(t, frame="base_link"):
    sec = int(np.floor(t))
    return T["std_msgs/msg/Header"](seq=0, stamp=T["builtin_interfaces/msg/Time"](sec=sec, nanosec=int((t - sec) * 1e9)),
                                    frame_id=frame)


def vec(x, y, z):
    return T["geometry_msgs/msg/Vector3"](x=float(x), y=float(y), z=float(z))


def odom_msg(t, x, y, vx, wz, frame="odom"):
    G = T
    pose = G["geometry_msgs/msg/PoseWithCovariance"](
        pose=G["geometry_msgs/msg/Pose"](position=G["geometry_msgs/msg/Point"](x=float(x), y=float(y), z=0.0),
                                         orientation=G["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0)),
        covariance=np.zeros(36))
    twist = G["geometry_msgs/msg/TwistWithCovariance"](
        twist=G["geometry_msgs/msg/Twist"](linear=vec(vx, 0, 0), angular=vec(0, 0, wz)), covariance=np.zeros(36))
    return G["nav_msgs/msg/Odometry"](header=hdr(t, frame), child_frame_id="base_link", pose=pose, twist=twist)


def imu_msg(t, gyro, acc):
    return T["sensor_msgs/msg/Imu"](
        header=hdr(t, "imu_link"),
        orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.zeros(9),
        angular_velocity=vec(*gyro), angular_velocity_covariance=np.zeros(9),
        linear_acceleration=vec(*acc), linear_acceleration_covariance=np.zeros(9))


def write_bag(path, topic, msgtype, items):
    path = Path(path)
    if path.exists():
        path.unlink()
    with Writer(path) as w:
        conn = w.add_connection(topic, msgtype, typestore=TS)
        for t, msg in items:
            w.write(conn, int(t * 1e9), TS.serialize_ros1(msg, msgtype))


def command_profile(rng, duration):
    """Orchard-like driving: straight rows (~1.2 m/s) and U-turns with varied turn rates."""
    dt = 0.02
    n = int(duration / dt)
    v = np.zeros(n)
    w = np.zeros(n)
    i = int(5 / dt)                     # 5 s stationary at start (gyro bias)
    while i < n:
        L = int(rng.uniform(12, 25) / dt)          # straight row
        vs = rng.uniform(1.0, 1.5)
        v[i:i + L] = vs + 0.05 * np.sin(np.arange(min(L, n - i)) * dt * 0.7)[:len(v[i:i + L])]
        w[i:i + L] = rng.normal(0, 0.02)
        i += L
        wt = rng.uniform(0.35, 1.3) * rng.choice([-1, 1])  # U-turn at varied rate
        Lt = int(np.pi / abs(wt) / dt)
        v[i:i + Lt] = rng.uniform(0.3, 0.8)
        w[i:i + Lt] = wt
        i += Lt
        if rng.random() < 0.15:                    # occasional short stop
            i += int(rng.uniform(1, 3) / dt)
    # smooth (velocity controller response ~0.15 s)
    ker = np.exp(-np.arange(0, 0.6, dt) / 0.15)
    ker /= ker.sum()
    v = np.convolve(v, ker)[:n]
    w = np.convolve(w, ker)[:n]
    return np.arange(n) * dt, v, w


def simulate(name, field, root, slip, seed, duration):
    rng = np.random.default_rng(seed)
    P = {"yes": FIELD_SLIP, "strong": STRONG_SLIP}[slip][field] if slip != "no" else NO_SLIP
    t, v_w, w_w = command_profile(rng, duration)
    t = t + 1.689e9 + seed * 5000.0
    dt = 0.02
    # true body velocities from the KNOWN slip model
    w_true = P["b"] * w_w - P["k"] * w_w * np.abs(w_w)
    vx = v_w * (P["a0"] - P["aw"] * np.abs(w_w))
    vy = P["c"] * w_w
    psi = np.cumsum(w_true) * dt + rng.uniform(-np.pi, np.pi)
    x = np.cumsum(np.cos(psi) * vx - np.sin(psi) * vy) * dt
    y = np.cumsum(np.sin(psi) * vx + np.cos(psi) * vy) * dt
    lx, ly = LEVER
    ax, ay = x + np.cos(psi) * lx - np.sin(psi) * ly, y + np.sin(psi) * lx + np.cos(psi) * ly

    d = Path(root) / name
    g = Path(root) / "ground_truth" / name
    d.mkdir(parents=True, exist_ok=True)
    g.mkdir(parents=True, exist_ok=True)
    stamp = f"2023-07-1{seed % 9}-14-00-00"

    # wheel odometry 50 Hz (what the wheels think)
    wo = [(ti, odom_msg(ti, 0, 0, vi + rng.normal(0, 0.005), wi + rng.normal(0, 0.005)))
          for ti, vi, wi in zip(t, v_w, w_w)]
    write_bag(d / f"odom_{stamp}.bag", "/jackal_velocity_controller/odom", "nav_msgs/msg/Odometry", wo)

    # IMU 200 Hz, mounted upside down + 90 deg yaw: R maps body->imu
    ti = np.arange(t[0], t[-1], 0.005)
    wti = np.interp(ti, t, w_true)
    Rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    Rx = np.diag([1.0, -1.0, -1.0])
    R = Rx @ Rz
    bias = np.array([0.003, -0.002, 0.004])
    gyro = (R @ np.vstack([0 * wti, 0 * wti, wti])).T + bias + rng.normal(0, 0.003, (len(ti), 3))
    acc = (R @ np.array([0, 0, 9.81])) + rng.normal(0, 0.05, (len(ti), 3))
    write_bag(d / f"base_{stamp}_0.bag", "/microstrain/imu/data", "sensor_msgs/msg/Imu",
              [(a, imu_msg(a, gg, aa)) for a, gg, aa in zip(ti, gyro, acc)])

    # RTK GPS 10 Hz at the antenna, 1 cm noise, stamps late by GPS_LAG; local UTM-like offset
    tg = t[::5]
    gx = ax[::5] + rng.normal(0, 0.01, len(tg)) + 4.6e5
    gy = ay[::5] + rng.normal(0, 0.01, len(tg)) + 3.76e6
    write_bag(g / "gt.bag", "/gps/fix/odometry", "nav_msgs/msg/Odometry",
              [(a + GPS_LAG, odom_msg(a + GPS_LAG, xx, yy, 0, 0, "utm")) for a, xx, yy in zip(tg, gx, gy)])
    with open(g / "gt.csv", "w") as fh:            # TUM-style csv as well (tests the csv path)
        for a, xx, yy in zip(tg, gx, gy):
            fh.write(f"{a + GPS_LAG:.6f} {xx:.4f} {yy:.4f} 0 0 0 0 1\n")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slip", choices=["yes", "strong", "no"], default="yes")
    ap.add_argument("--duration", type=float, default=420.0)
    a = ap.parse_args()
    drives = {"01_13B_Jackal": "13B", "02_13B_Jackal": "13B", "04_13D_Jackal": "13D", "05_13D_Jackal": "13D"}
    truth = {}
    for k, (name, field) in enumerate(drives.items()):
        truth[name] = simulate(name, field, a.out, a.slip, seed=k + 1, duration=a.duration)
        print("wrote", name)
    cfg = dict(data_root=os.path.abspath(a.out), drives=drives, lever_arm_xy=list(LEVER),
               imu_preference=["microstrain", "zed"], output_dir=os.path.join(os.path.abspath(a.out), "results"))
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump(dict(slip_params=truth, lever=LEVER, gps_lag=GPS_LAG), open(os.path.join(a.out, "truth.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
