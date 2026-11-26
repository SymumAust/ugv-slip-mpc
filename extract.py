#!/usr/bin/env python3
"""
Extract only the signals this project uses from the raw CitrusFarm bags into small, git-friendly files:

    data/processed/<drive>/odom.csv.gz   t, v_w, w_w                       (50 Hz wheel odometry)
    data/processed/<drive>/imu.csv.gz    t, gx, gy, gz, ax, ay, az         (IMU, raw sensor frame)
    data/processed/<drive>/gt.csv.gz     t, x, y                           (RTK GPS antenna, as recorded)
    data/processed/<drive>/meta.json     source files, IMU source, row counts

The raw bags (many GB) stay outside git. The extracted files are a few MB per drive.
No processing is applied here (no sync, no filtering), so every later step stays reproducible from these files.

Usage:
    python extract.py --config config.json            # uses data_root and drives from config
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import feasibility as fz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    root = cfg["data_root"]
    for name in cfg["drives"]:
        drive_dir = os.path.join(root, name)
        gt_dir = os.path.join(root, "ground_truth", name)
        if not os.path.isdir(drive_dir):
            fz.log(f"{name}: raw folder not found, skipped")
            continue
        flags = []
        try:
            t_o, v_o, w_o = fz.load_odom(drive_dir, flags)
            t_i, gyro, acc, src = fz.load_imu(drive_dir, cfg.get("imu_preference", ["microstrain", "zed"]), flags)
            t_g, x_g, y_g, gt_src = fz.load_gt(gt_dir, flags, local=False)
        except FileNotFoundError as e:
            fz.log(f"{name}: SKIPPED, missing input: {e}")
            continue
        od = os.path.join(args.out, name)
        os.makedirs(od, exist_ok=True)
        pd.DataFrame({"t": t_o, "v_w": v_o, "w_w": w_o}).to_csv(
            os.path.join(od, "odom.csv.gz"), index=False, float_format="%.6f")
        pd.DataFrame(np.column_stack([t_i, gyro, acc]), columns=["t", "gx", "gy", "gz", "ax", "ay", "az"]).to_csv(
            os.path.join(od, "imu.csv.gz"), index=False, float_format="%.6f")
        pd.DataFrame({"t": t_g, "x": x_g, "y": y_g}).to_csv(
            os.path.join(od, "gt.csv.gz"), index=False, float_format="%.4f")
        meta = dict(drive=name, field=cfg["drives"][name], imu_source=src, gt_source=gt_src,
                    rows=dict(odom=len(t_o), imu=len(t_i), gt=len(t_g)), load_flags=flags,
                    note="Extracted from CitrusFarm (CC BY-SA 4.0, Teng et al. 2023). Timestamps: header stamps, s.")
        json.dump(meta, open(os.path.join(od, "meta.json"), "w"), indent=2)
        size = sum(os.path.getsize(os.path.join(od, f)) for f in os.listdir(od)) / 1e6
        fz.log(f"{name}: odom {len(t_o)}, imu[{src}] {len(t_i)}, gt {len(t_g)} rows -> {od} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
