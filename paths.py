#!/usr/bin/env python3
"""
Step 1: reference paths and corridor from the CitrusFarm RTK-GPS tracks.

For each drive:
  1. load GPS (antenna position), resample every DS metres of travelled distance;
  2. smooth (Savitzky-Golay over ~2 m) and compute heading and curvature;
  3. split into ROW passes and TURN segments (90-degree corners + headland transit), in two
     passes: (a) clean rows = long stretches heading along the row axis; they define the lanes
     (centre + ends); (b) every point inside a lane, between its ends and heading within 45
     degrees of the row axis is a row point, so an operator weaving inside a lane is still a row;
  4. replace each row pass by a straight line (rows in an orchard are straight; the line's
     lateral offset = mean lateral position of the pass = estimated lane centre);
  5. lightly re-smooth the whole path so row/turn junctions have no kinks, resample, and
     recompute heading psi and curvature kappa (the network's preview input later).
Across drives:
  6. estimate the row (lane) spacing from the offsets of adjacent row passes;
  7. report how far the human operator strayed from the lane centre (a data-based lower
     bound on the free lane width) -> basis for the corridor half-width d.

Outputs:
  data/paths/<drive>.csv      s, x, y, psi, kappa, seg (row/turn), seg_id
  results/paths/summary.json  spacing, deviation statistics, segment counts
  results/paths/paths.png     map of every drive (rows blue, turns red, raw GPS grey)
  results/paths/kappa.png     curvature along the path of one drive
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DS = 0.10                 # resampling step along the path (m)
SMOOTH_M = 5.1            # smoothing window for turns (m); chosen from 2.1/4.1/5.1 by lowest curvature spikes at equal GPS fit
JOIN_SMOOTH_M = 5.1       # second, light smoothing that removes kinks at row/turn joins (m)
ROW_ANGLE_DEG = 15.0      # a point is "along the row" if its heading is within this of the row axis
CLASSIFY_SMOOTH_M = 6.1   # heading used for the row/turn decision is smoothed over this distance (m),
                          # so an operator weaving inside a lane is not mistaken for a turn
ROW_MIN_LEN = 15.0        # a row pass must be at least this long (m)
BLEND_M = 4.0             # at each end of a row pass the reference moves gradually from the lane
                          # centre to the driven track over this distance, so rows meet turns without a step
EDGE_TRIM = 3.0           # ignore this much at each end of a row pass for offset/deviation stats (m)
MERGE_OFFSET = 1.5        # row offsets closer than this are the same lane (repeated pass)
LANE_TOL = 1.5            # pass 2: a point within this of a lane centre (across the row) belongs to that lane
LANE_END_MARGIN = 1.0     # pass 2: and at least this far inside the lane's ends (along the row)
LANE_HEADING_DEG = 45.0   # pass 2: and heading within this of the row axis (allows weaving, not turning)
FIELD = {"01": "13B", "02": "13B", "03": "13B", "04": "13D", "05": "13D"}

OUT_PATHS, OUT_RES = "data/paths", "results/paths"


def load_gps(f):
    d = pd.read_csv(f, comment="#", header=None, names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
    return d.t.to_numpy(), d.x.to_numpy(), d.y.to_numpy()


def resample_by_distance(x, y, ds=DS):
    step = np.hypot(np.diff(x), np.diff(y))
    keep = np.r_[True, step > 1e-4]                     # drop repeated positions (robot standing)
    x, y = x[keep], y[keep]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    su = np.arange(0, s[-1], ds)
    return su, np.interp(su, s, x), np.interp(su, s, y)


def smooth_geometry(x, y, window_m, ds=DS):
    w = int(round(window_m / ds)) | 1                   # odd window in samples
    xs, ys = savgol_filter(x, w, 3), savgol_filter(y, w, 3)
    dx, dy = savgol_filter(x, w, 3, deriv=1, delta=ds), savgol_filter(y, w, 3, deriv=1, delta=ds)
    ddx, ddy = savgol_filter(x, w, 3, deriv=2, delta=ds), savgol_filter(y, w, 3, deriv=2, delta=ds)
    psi = np.unwrap(np.arctan2(dy, dx))
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
    return xs, ys, psi, kappa


def row_axis(psi, kappa):
    """Dominant row direction (mod pi) from straight parts of the path."""
    straight = np.abs(kappa) < 0.02
    a = 2 * psi[straight]                               # doubling angles makes theta and theta+pi agree
    return 0.5 * np.arctan2(np.sin(a).mean(), np.cos(a).mean())


def runs(mask):
    """Start/end indices (end exclusive) of True runs."""
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(e[::2], e[1::2]))


def process_drive(f):
    name = os.path.basename(f).replace(".csv.gz", "")
    _, xr, yr = load_gps(f)
    s, x, y = resample_by_distance(xr, yr)
    xs, ys, psi, kappa = smooth_geometry(x, y, SMOOTH_M)
    theta = row_axis(psi, kappa)
    u = np.array([np.cos(theta), np.sin(theta)])        # along-row unit vector
    n = np.array([-np.sin(theta), np.cos(theta)])       # across-row unit vector

    # pass 1: clean row passes = long stretches heading along the row axis
    _, _, psi_c, _ = smooth_geometry(x, y, CLASSIFY_SMOOTH_M)
    along = np.abs(np.sin(psi_c - theta)) < np.sin(np.radians(ROW_ANGLE_DEG))
    clean = [(a, b) for a, b in runs(along) if (b - a) * DS >= ROW_MIN_LEN]
    if not clean:
        raise RuntimeError(f"{name}: no row passes found")
    # lanes (across-row centre + along-row extent) from the clean passes
    lat_all, lon_all = np.column_stack([xs, ys]) @ n, np.column_stack([xs, ys]) @ u
    lanes = []
    for a, b in sorted(clean, key=lambda r: lat_all[r[0]:r[1]].mean()):
        c = lat_all[a:b].mean()
        if lanes and c - lanes[-1]["c"][-1] < MERGE_OFFSET:
            lanes[-1]["c"].append(c); lanes[-1]["lo"].append(lon_all[a:b].min()); lanes[-1]["hi"].append(lon_all[a:b].max())
        else:
            lanes.append(dict(c=[c], lo=[lon_all[a:b].min()], hi=[lon_all[a:b].max()]))
    # lanes lie on a regular grid: a gap of k spacings between two known lanes hides k-1 lanes that
    # the robot drove only while weaving (no clean pass) -> insert them, with the neighbours' extent
    cs = np.array([np.mean(L["c"]) for L in lanes])
    gaps = np.diff(cs)
    if np.any(gaps < 10):
        sp = np.median(gaps[gaps < 10])
        filled = [lanes[0]]
        for L0, L1, g in zip(lanes[:-1], lanes[1:], gaps):
            k = int(round(g / sp))
            if 2 <= k <= 3 and abs(g - k * sp) < 0.5 * MERGE_OFFSET:
                for j in range(1, k):
                    filled.append(dict(c=[np.mean(L0["c"]) + j * g / k], lo=[max(min(L0["lo"]), min(L1["lo"]))],
                                       hi=[min(max(L0["hi"]), max(L1["hi"]))], inferred=True))
            filled.append(L1)
        lanes = filled
    # pass 2: any point inside a lane, between its ends, heading roughly along the row, is a row point
    in_row = np.zeros(len(s), bool)
    head_ok = np.abs(np.sin(psi - theta)) < np.sin(np.radians(LANE_HEADING_DEG))
    # a lane's ends: its own clean passes, widened to the field's typical lane ends (lanes in one
    # field are about equally long; a lane driven only partly cleanly would otherwise be cut short)
    med_lo, med_hi = np.median([min(L["lo"]) for L in lanes]), np.median([max(L["hi"]) for L in lanes])
    for L in lanes:
        c, lo, hi = np.mean(L["c"]), min(min(L["lo"]), med_lo), max(max(L["hi"]), med_hi)
        in_row |= (np.abs(lat_all - c) < LANE_TOL) & (lon_all > lo + LANE_END_MARGIN) & (lon_all < hi - LANE_END_MARGIN) & head_ok
    rows = [(a, b) for a, b in runs(in_row) if (b - a) * DS >= ROW_MIN_LEN]
    first, last = rows[0][0], rows[-1][1]               # trim start/end manoeuvres outside the rows

    seg = np.full(len(s), "turn", dtype=object)
    seg_id = np.full(len(s), -1)
    xref, yref = xs.copy(), ys.copy()
    offsets, devs, row_info = [], [], []
    k = int(EDGE_TRIM / DS)
    for i, (a, b) in enumerate(rows):
        P = np.column_stack([x[a:b], y[a:b]])            # raw (resampled, unsmoothed) GPS of the pass
        lat = P @ n
        off = lat[k:-k].mean() if b - a > 2 * k + 10 else lat.mean()
        offsets.append(off)
        devs.append(lat[k:-k] - off)
        lon = P @ u
        # lateral position of the reference: the lane centre, blending into the smoothed driven
        # track over BLEND_M at both ends of the pass (w = 1 on the lane centre, 0 at the pass ends)
        lat_drv = np.column_stack([xs[a:b], ys[a:b]]) @ n
        m = b - a
        pos = np.arange(m) * DS
        w = np.clip(np.minimum(pos, (m - 1) * DS - pos) / BLEND_M, 0, 1)
        w = 0.5 - 0.5 * np.cos(np.pi * w)                 # smooth (cosine) ramp
        lat_ref = w * off + (1 - w) * lat_drv
        xref[a:b], yref[a:b] = lon * u[0] + lat_ref * n[0], lon * u[1] + lat_ref * n[1]
        seg[a:b], seg_id[a:b] = "row", i
        row_info.append(dict(id=i, length_m=round((b - a) * DS, 1), offset_m=round(float(off), 3),
                             dev_std_m=round(float(np.std(lat[k:-k] - off)), 3)))

    # turn segments get their own ids, numbered after the rows
    t_id = len(rows)
    for a, b in runs(seg == "turn"):
        seg_id[a:b] = t_id
        t_id += 1

    sl = slice(first, last)
    xref, yref, seg, seg_id = xref[sl], yref[sl], seg[sl], seg_id[sl]
    # light second smoothing removes the small steps where a straight row line meets a turn
    s2, xr2, yr2 = resample_by_distance(xref, yref)
    xf, yf, psif, kappaf = smooth_geometry(xr2, yr2, JOIN_SMOOTH_M)
    # carry the row/turn labels over to the resampled path by nearest original index
    src = np.r_[0, np.cumsum(np.hypot(np.diff(xref), np.diff(yref)))]
    idx = np.clip(np.searchsorted(src, s2), 0, len(seg) - 1)
    df = pd.DataFrame(dict(s=np.round(s2, 3), x=np.round(xf, 4), y=np.round(yf, 4), psi=np.round(psif, 5),
                           kappa=np.round(kappaf, 5), seg=seg[idx], seg_id=seg_id[idx]))

    # how far the reference ended up from the raw GPS (sanity check of the smoothing)
    tree_ref = np.column_stack([xf, yf])
    raw = np.column_stack([x[sl], y[sl]])[::5]
    d_ref = np.min(np.hypot(raw[:, None, 0] - tree_ref[None, ::5, 0], raw[:, None, 1] - tree_ref[None, ::5, 1]), axis=1)

    info = dict(drive=name, field=FIELD[name[:2]], row_axis_deg=round(float(np.degrees(theta)), 2),
                path_length_m=round(float(s2[-1]), 1), n_rows=len(rows),
                row_len_median_m=round(float(np.median([r["length_m"] for r in row_info])), 1),
                kappa_abs_p99_turns=round(float(np.percentile(np.abs(kappaf[df.seg == "turn"]), 99)), 3),
                ref_to_raw_gps_median_m=round(float(np.median(d_ref)), 3),
                ref_to_raw_gps_p99_m=round(float(np.percentile(d_ref, 99)), 3),
                rows=row_info)
    return name, df, dict(x=x, y=y), info, np.array(offsets), np.concatenate(devs)


def lane_spacing(offsets):
    """Spacing between adjacent lanes: merge repeated passes, sort, take consecutive differences."""
    o = np.sort(offsets)
    lanes = [[o[0]]]
    for v in o[1:]:
        if v - lanes[-1][-1] < MERGE_OFFSET:
            lanes[-1].append(v)          # same lane driven again
        else:
            lanes.append([v])            # next lane
    centres = np.array([np.mean(lane) for lane in lanes])
    return np.diff(centres)


def main():
    os.makedirs(OUT_PATHS, exist_ok=True)
    os.makedirs(OUT_RES, exist_ok=True)
    results = {}
    for f in sorted(glob.glob("data/gt/*.csv.gz")):
        name, df, raw, info, offs, devs = process_drive(f)
        df.to_csv(os.path.join(OUT_PATHS, f"{name}.csv"), index=False)
        diffs = lane_spacing(offs)
        info["lane_gaps_m"] = [round(float(v), 2) for v in diffs]
        results[name] = dict(df=df, raw=raw, info=info, devs=devs, diffs=diffs)
        print(f"{name}: {info['n_rows']} rows, path {info['path_length_m']} m, row axis {info['row_axis_deg']} deg, "
              f"ref-to-GPS median {info['ref_to_raw_gps_median_m']} m, lane gaps {info['lane_gaps_m']}")

    # ---- lane spacing per field, from drives that visit adjacent lanes (gaps below 10 m)
    summary = dict(per_drive={k: {kk: vv for kk, vv in r["info"].items() if kk != "rows"} for k, r in results.items()})
    for fld in ("13B", "13D"):
        gaps = np.concatenate([r["diffs"][r["diffs"] < 10] for r in results.values() if r["info"]["field"] == fld])
        sp = float(np.median(gaps))
        devs = np.concatenate([r["devs"] for r in results.values() if r["info"]["field"] == fld])
        summary[fld] = dict(
            lane_spacing_m=round(sp, 2), lane_spacing_iqr_m=[round(float(v), 2) for v in np.percentile(gaps, [25, 75])],
            n_gaps=int(len(gaps)),
            operator_dev_std_m=round(float(np.std(devs)), 3),
            operator_dev_abs_p95_m=round(float(np.percentile(np.abs(devs), 95)), 3),
            operator_dev_abs_p99_m=round(float(np.percentile(np.abs(devs), 99)), 3),
            operator_dev_abs_max_m=round(float(np.max(np.abs(devs))), 3))
        # gaps of drives that skip lanes, expressed in lane spacings (should be close to whole numbers)
        summary[fld]["wide_gaps_in_spacings"] = {
            k: [round(float(v / sp), 2) for v in r["diffs"] if v >= 10]
            for k, r in results.items() if r["info"]["field"] == fld and np.any(r["diffs"] >= 10)}
        print(f"field {fld}: lane spacing {sp:.2f} m (IQR {summary[fld]['lane_spacing_iqr_m']}, n={len(gaps)}); "
              f"operator |deviation| p95 {summary[fld]['operator_dev_abs_p95_m']} m, "
              f"p99 {summary[fld]['operator_dev_abs_p99_m']} m, max {summary[fld]['operator_dev_abs_max_m']} m; "
              f"wide gaps in spacings {summary[fld]['wide_gaps_in_spacings']}")
    json.dump(summary, open(os.path.join(OUT_RES, "summary.json"), "w"), indent=2)
    json.dump({k: r["info"]["rows"] for k, r in results.items()}, open(os.path.join(OUT_RES, "rows.json"), "w"), indent=1)
    figures(results)


def figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(22, 8))
    for a, (name, r) in zip(ax, results.items()):
        df = r["df"]
        a.plot(r["raw"]["x"], r["raw"]["y"], color="0.75", lw=2.2, label="raw GPS")
        for sid, g in df.groupby("seg_id"):
            a.plot(g.x, g.y, color="tab:blue" if g.seg.iloc[0] == "row" else "tab:red", lw=0.9)
        a.plot([], [], color="tab:blue", label="row (reference)")
        a.plot([], [], color="tab:red", label="turn (reference)")
        a.set_aspect("equal")
        a.set_title(f"{name}\n{r['info']['n_rows']} rows, {r['info']['path_length_m']:.0f} m", fontsize=10)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Reference paths from CitrusFarm RTK-GPS: straight rows (blue), smoothed turns (red), raw GPS (grey)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_RES, "paths.png"), dpi=110)
    plt.close(fig)

    name = "04_13D_Jackal" if "04_13D_Jackal" in results else next(iter(results))
    df = results[name]["df"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df.s, df.kappa, lw=0.8, color="k")
    turn = (df.seg == "turn").to_numpy()
    ax.fill_between(df.s, -2, 2, where=turn, color="tab:red", alpha=0.12, step="mid", label="turn segment")
    ax.set_ylim(-1.2, 1.2)
    ax.set(xlabel="distance along path s (m)", ylabel="curvature κ (1/m)", title=f"Curvature along the reference path, {name}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_RES, "kappa.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
