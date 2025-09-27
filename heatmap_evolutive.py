#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Evolving Heatmap of Movements (SQL/Python stack) with hour filtering and segments intensity

import argparse
import logging
import time
import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib import cm

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ----------------------
# Logging helpers
# ----------------------
def setup_logging(verbosity: int):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------
# Utility: separable 2D smoothing (no SciPy), avoids shape-mismatch
# ----------------------
def _conv1d_along_axis(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        ap = np.pad(a, ((pad, pad),(0,0)), mode="edge")
        out = np.empty_like(a, dtype=float)
        for j in range(a.shape[1]):
            out[:, j] = np.convolve(ap[:, j], kernel, mode="valid")
        return out
    elif axis == 1:
        ap = np.pad(a, ((0,0),(pad, pad)), mode="edge")
        out = np.empty_like(a, dtype=float)
        for i in range(a.shape[0]):
            out[i, :] = np.convolve(ap[i, :], kernel, mode="valid")
        return out
    else:
        raise ValueError("axis must be 0 or 1")

def smooth2d(arr: np.ndarray, ksize: int = 3, passes: int = 1) -> np.ndarray:
    if ksize < 2 or passes < 1:
        return arr
    if ksize % 2 == 0:
        ksize += 1  # enforce odd
    kernel = np.ones(ksize, dtype=float) / float(ksize)
    out = arr.astype(float, copy=True)
    for _ in range(passes):
        out = _conv1d_along_axis(out, kernel, axis=0)
        out = _conv1d_along_axis(out, kernel, axis=1)
    return out

# ----------------------
# Load & prepare
# ----------------------
def load_data(beacons_csv: str, segments_csv: str, counts_csv: str, hour_min: int, hour_max: int):
    t0 = time.time()
    logging.info(f"Reading CSVs...")
    beacons = pd.read_csv(beacons_csv, dtype=str)
    segments = pd.read_csv(segments_csv, dtype=str)
    counts = pd.read_csv(counts_csv, dtype=str, low_memory=False)
    logging.info(f"Read beacons={len(beacons):,}, segments={len(segments):,}, counts={len(counts):,} rows in {time.time()-t0:.2f}s")

    # Normalize column names
    beacons.columns = [c.strip().lower() for c in beacons.columns]
    segments.columns = [c.strip().lower() for c in segments.columns]
    counts.columns = [c.strip().lower() for c in counts.columns]

    required_beacons = {"id","x_m","y_m","floor"}
    required_segments = {"segment_id","x1_m","x2_m","y1_m","y2_m","floor"}
    required_counts = {"floor","beacon","date_it","hour","count"}

    for req, df, name in [
        (required_beacons, beacons, "beacons"),
        (required_segments, segments, "segments"),
        (required_counts, counts, "counts"),
    ]:
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {name} CSV: {missing}")

    # Coerce numeric columns safely
    for col in ["x_m","y_m","floor"]:
        beacons[col] = pd.to_numeric(beacons[col], errors="coerce")
    for col in ["x1_m","x2_m","y1_m","y2_m","floor"]:
        segments[col] = pd.to_numeric(segments[col], errors="coerce")
    for col in ["floor","hour","count"]:
        counts[col] = pd.to_numeric(counts[col], errors="coerce")

    # Basic cleaning
    beacons = beacons.dropna(subset=["id","x_m","y_m","floor"])
    segments = segments.dropna(subset=["segment_id","x1_m","x2_m","y1_m","y2_m","floor"])
    counts = counts.dropna(subset=["beacon","floor","date_it","hour","count"])

    # Cast types
    beacons["floor"] = beacons["floor"].astype(int)
    segments["floor"] = segments["floor"].astype(int)
    counts["floor"] = counts["floor"].astype(int)
    counts["hour"] = counts["hour"].astype(int)
    counts["count"] = counts["count"].astype(float)

    # Filter on hour window (inclusive)
    if hour_min is not None and hour_max is not None:
        logging.info(f"Filtering counts on hour between {hour_min} and {hour_max} inclusive")
        pre = len(counts)
        counts = counts[(counts["hour"] >= hour_min) & (counts["hour"] <= hour_max)]
        logging.info(f"Filtered {pre - len(counts):,} rows outside hour window; remaining={len(counts):,}")

    # Timestamp as pandas datetime (date_it + hour)
    counts["ts"] = pd.to_datetime(counts["date_it"], errors="coerce") + pd.to_timedelta(counts["hour"], unit="h")
    pre = len(counts)
    counts = counts.dropna(subset=["ts"])
    logging.info(f"Dropped {pre - len(counts):,} count rows with invalid timestamps; remaining={len(counts):,}")

    return beacons, segments, counts

# ----------------------
# Build frames (beacons heatmap)
# ----------------------
def make_frames_heatmap(beacons: pd.DataFrame, counts: pd.DataFrame, floor: int,
                        bins: int = 100, smooth_kernel: int = 3, smooth_passes: int = 1):
    b_floor = beacons[beacons["floor"] == floor].copy()
    if b_floor.empty:
        raise ValueError(f"No beacons found for floor {floor}.")
    # Join counts to beacon coords
    c_floor = counts[counts["floor"] == floor].merge(
        b_floor.rename(columns={"id":"beacon"}), on=["beacon","floor"], how="inner"
    )
    if c_floor.empty:
        raise ValueError(f"No counts joined with beacons for floor {floor}.")

    logging.info(f"Floor {floor}: beacons={len(b_floor):,}, joined counts={len(c_floor):,}")
    # Canvas bounds
    xmin, xmax = b_floor["x_m"].min(), b_floor["x_m"].max()
    ymin, ymax = b_floor["y_m"].min(), b_floor["y_m"].max()
    # Pad bounds a bit
    pad_x = max(1.0, 0.05*(xmax - xmin))
    pad_y = max(1.0, 0.05*(ymax - ymin))
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    # Time order
    times = np.sort(c_floor["ts"].unique())

    frames = []
    iterator = tqdm(times, desc=f"Frames heatmap (floor {floor})") if _HAS_TQDM else times
    for ts in iterator:
        df_t = c_floor[c_floor["ts"] == ts]
        # Histogram2d weighted by count
        H, xedges, yedges = np.histogram2d(
            df_t["x_m"].values.astype(float), df_t["y_m"].values.astype(float),
            bins=bins, range=[[xmin, xmax],[ymin, ymax]],
            weights=df_t["count"].values.astype(float),
        )
        H = H.T  # align to (y,x) for imshow
        if smooth_kernel and smooth_kernel > 1:
            H = smooth2d(H, ksize=int(smooth_kernel), passes=int(smooth_passes))
        frames.append((ts, H, (xmin,xmax,ymin,ymax)))
    logging.info(f"Built {len(frames)} heatmap frames for floor {floor}.")
    return frames

# ----------------------
# Geometry helpers
# ----------------------
def _point_to_segment_distance(px, py, x1, y1, x2, y2) -> float:
    # Compute distance from point P to segment AB
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx*wx + vy*wy
    if c1 <= 0:
        return np.hypot(px - x1, py - y1)
    c2 = vx*vx + vy*vy
    if c2 <= c1:
        return np.hypot(px - x2, py - y2)
    b = c1 / c2
    bx, by = x1 + b*vx, y1 + b*vy
    return np.hypot(px - bx, py - by)

# Precompute beacon-to-segment neighborhood within a radius
def precompute_segment_neighbors(b_floor: pd.DataFrame, seg_f: pd.DataFrame, radius: float):
    neighbors = []
    b_floor = b_floor.reset_index(drop=True)
    bxy = b_floor[["x_m","y_m"]].to_numpy(float)
    for _, s in seg_f.iterrows():
        x1, x2, y1, y2 = float(s["x1_m"]), float(s["x2_m"]), float(s["y1_m"]), float(s["y2_m"])
        wlist = []
        for bi, (bx, by) in enumerate(bxy):
            d = _point_to_segment_distance(bx, by, x1, y1, x2, y2)
            if d <= radius:
                w = 1.0 / (1.0 + d)  # inverse-distance weight
                wlist.append((bi, w))
        neighbors.append(wlist)
    return neighbors

# ----------------------
# Build frames (segments intensity from nearby beacons)
# ----------------------
def make_frames_segments(beacons: pd.DataFrame, segments: pd.DataFrame, counts: pd.DataFrame, floor: int,
                         radius: float = 5.0):
    b_floor = beacons[beacons["floor"] == floor].reset_index(drop=True).copy()
    seg_f = segments[segments["floor"] == floor].reset_index(drop=True).copy()
    if b_floor.empty or seg_f.empty:
        raise ValueError(f"Need beacons and segments on floor {floor}.")
    logging.info(f"Floor {floor}: computing segment neighbors within radius={radius}m")
    neighbors = precompute_segment_neighbors(b_floor, seg_f, radius=radius)

    # Join counts to beacons to get time series per beacon
    c_floor = counts[counts["floor"] == floor].merge(
        b_floor.rename(columns={"id":"beacon"}), on=["beacon","floor"], how="inner"
    )
    if c_floor.empty:
        raise ValueError(f"No counts joined with beacons for floor {floor}.")

    # distinct times
    times = np.sort(c_floor["ts"].unique())

    # Prepare line segments for plotting
    seg_lines = np.stack([
        np.array([[row["x1_m"], row["y1_m"]],[row["x2_m"], row["y2_m"]]], dtype=float)
        for _, row in seg_f.iterrows()
    ], axis=0)

    frames = []
    iterator = tqdm(times, desc=f"Frames segments (floor {floor})") if _HAS_TQDM else times
    for ts in iterator:
        df_t = c_floor[c_floor["ts"] == ts]
        # Map from beacon id to count at ts
        counts_map = dict(zip(df_t["beacon"].astype(str), df_t["count"].astype(float)))
        # Ensure beacon id column name
        beacon_col = "beacon" if "beacon" in b_floor.columns else "id"
        b_counts = np.array([counts_map.get(str(bid), 0.0) for bid in b_floor[beacon_col]], dtype=float)

        seg_values = np.zeros(len(seg_f), dtype=float)
        for si, wlist in enumerate(neighbors):
            if not wlist:
                continue
            val = 0.0; wsum = 0.0
            for bi, w in wlist:
                val += w * b_counts[bi]
                wsum += w
            seg_values[si] = val / wsum if wsum > 0 else 0.0

        frames.append((ts, seg_lines, seg_values))
    logging.info(f"Built {len(frames)} segment frames for floor {floor}.")
    return frames

# ----------------------
# Plot & animate
# ----------------------
def render_animation_heatmap(frames, segments: pd.DataFrame, floor: int, out_path: str, fps: int = 6, cmap: str = "viridis"):
    if not frames:
        raise ValueError("No frames to render.")
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    # Draw segments for this floor as context (thin gray)
    seg_f = segments[segments["floor"] == floor]
    for _, s in seg_f.iterrows():
        ax.plot([s["x1_m"], s["x2_m"]], [s["y1_m"], s["y2_m"]], linewidth=1.0, alpha=0.6, color="black")

    (ts0, H0, (xmin,xmax,ymin,ymax)) = frames[0]
    im = ax.imshow(
        H0, origin="lower", extent=[xmin, xmax, ymin, ymax],
        interpolation="nearest", cmap=cmap, aspect="equal"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity (counts)")
    title = ax.set_title(f"Floor {floor} — {pd.to_datetime(ts0).strftime('%Y-%m-%d %H:%M')}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])

    vmax = max(float(np.nanmax(H)) for _, H, _ in frames if H is not None)
    if vmax <= 0: vmax = 1.0
    im.set_clim(0, vmax)

    def update(i):
        ts, H, _ = frames[i]
        im.set_data(H)
        title.set_text(f"Floor {floor} — {pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    out_path = _save_anim(anim, out_path, fps)
    plt.close(fig)
    logging.info(f"Rendered heatmap to {out_path} in {time.time()-t0:.2f}s")
    return out_path

def render_animation_segments(frames, out_path: str, fps: int = 6, cmap_name: str = "viridis"):
    if not frames:
        raise ValueError("No frames to render.")
    t0 = time.time()
    # Bounds
    seg_lines = frames[0][1]
    xmin = np.min(seg_lines[:,:,0]); xmax = np.max(seg_lines[:,:,0])
    ymin = np.min(seg_lines[:,:,1]); ymax = np.max(seg_lines[:,:,1])
    pad_x = max(1.0, 0.05*(xmax - xmin)); pad_y = max(1.0, 0.05*(ymax - ymin))
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax]); ax.set_aspect("equal")

    lc = LineCollection(frames[0][1], linewidths=1.0)
    cmap = cm.get_cmap(cmap_name)
    vmax = max(np.nanmax(vals) for _, _, vals in frames)
    if vmax <= 0: vmax = 1.0
    colors = cmap(np.clip(frames[0][2] / vmax, 0, 1))
    lc.set_color(colors)
    ax.add_collection(lc)
    cbar = plt.colorbar(cm.ScalarMappable(norm=plt.Normalize(0,vmax), cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Segment intensity (weighted by nearby beacons)")
    title = ax.set_title(f"{pd.to_datetime(frames[0][0]).strftime('%Y-%m-%d %H:%M')}")

    def update(i):
        ts, lines, vals = frames[i]
        lc.set_segments(lines)
        colors = cmap(np.clip(vals / vmax, 0, 1))
        widths = 0.5 + 3.0 * (np.clip(vals / vmax, 0, 1))
        lc.set_color(colors)
        lc.set_linewidths(widths)
        title.set_text(f"{pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')}")
        return [lc, title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    out_path = _save_anim(anim, out_path, fps)
    plt.close(fig)
    logging.info(f"Rendered segment animation to {out_path} in {time.time()-t0:.2f}s")
    return out_path

def render_animation_both(frames_heat, frames_seg, segments, floor: int, out_path: str, fps: int = 6, cmap_name: str = "viridis"):
    if not frames_heat or not frames_seg:
        raise ValueError("Need both heatmap and segments frames.")
    t0 = time.time()
    (ts0, H0, (xmin,xmax,ymin,ymax)) = frames_heat[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(H0, origin="lower", extent=[xmin, xmax, ymin, ymax],
                   interpolation="nearest", cmap=cmap_name, aspect="equal")
    # static background segments
    seg_f = segments[segments["floor"] == floor]
    for _, s in seg_f.iterrows():
        ax.plot([s["x1_m"], s["x2_m"]], [s["y1_m"], s["y2_m"]], linewidth=0.6, alpha=0.4, color="black")

    lc = LineCollection(frames_seg[0][1], linewidths=1.0)
    cmap = cm.get_cmap(cmap_name)
    vmax_hm = max(float(np.nanmax(H)) for _, H, _ in frames_heat if H is not None)
    vmax_seg = max(np.nanmax(vals) for _, _, vals in frames_seg)
    vmax_hm = max(vmax_hm, 1.0); vmax_seg = max(vmax_seg, 1.0)
    im.set_clim(0, vmax_hm)
    colors = cmap(np.clip(frames_seg[0][2] / vmax_seg, 0, 1))
    lc.set_color(colors)
    ax.add_collection(lc)

    cbar1 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar1.set_label("Beacons heat intensity")
    cbar2 = plt.colorbar(cm.ScalarMappable(norm=plt.Normalize(0,vmax_seg), cmap=cmap), ax=ax, fraction=0.046, pad=0.04); cbar2.set_label("Segment intensity")

    title = ax.set_title(f"Floor {floor} — {pd.to_datetime(ts0).strftime('%Y-%m-%d %H:%M')}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])

    def update(i):
        ts_h, H, _ = frames_heat[i]
        ts_s, lines, vals = frames_seg[i]
        im.set_data(H)
        lc.set_segments(lines)
        colors = cmap(np.clip(vals / vmax_seg, 0, 1))
        widths = 0.5 + 3.0 * (np.clip(vals / vmax_seg, 0, 1))
        lc.set_color(colors)
        lc.set_linewidths(widths)
        title.set_text(f"Floor {floor} — {pd.to_datetime(ts_h).strftime('%Y-%m-%d %H:%M')}")
        return [im, lc, title]

    frames = min(len(frames_heat), len(frames_seg))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
    out_path = _save_anim(anim, out_path, fps)
    plt.close(fig)
    logging.info(f"Rendered combined animation to {out_path} in {time.time()-t0:.2f}s")
    return out_path

def _save_anim(anim, out_path: str, fps: int) -> str:
    out_path = str(out_path)
    if out_path.lower().endswith(".mp4"):
        try:
            writer = FFMpegWriter(fps=fps, metadata={'artist':'heatmap_evolutive'})
            anim.save(out_path, writer=writer, dpi=150)
        except Exception as e:
            logging.warning(f"FFMpeg failed, falling back to GIF: {e}")
            gif_path = out_path.rsplit(".",1)[0] + ".gif"
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer, dpi=150)
            out_path = gif_path
    else:
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=150)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Build an evolving heatmap from beacon movements.")
    ap.add_argument("--beacons", type=str, default="beacons_world_template.csv", help="Path to beacons CSV")
    ap.add_argument("--segments", type=str, default="segments_template.csv", help="Path to segments CSV")
    ap.add_argument("--counts", type=str, default="porta_di_roma_counts.csv", help="Path to counts CSV")
    ap.add_argument("--out", type=str, default="heatmap_floor0.gif", help="Output file (.gif or .mp4)")
    ap.add_argument("--fps", type=int, default=6, help="Frames per second")
    ap.add_argument("--bins", type=int, default=120, help="Number of histogram bins (resolution)")
    ap.add_argument("--smooth-kernel", type=int, default=3, help="Kernel size for smoothing (odd int >= 1). Use 1 to disable.")
    ap.add_argument("--smooth-passes", type=int, default=1, help="Number of smoothing passes")
    ap.add_argument("--floor", type=str, default="0", help="'all' or a specific floor integer, e.g. 0")
    ap.add_argument("--hour-min", type=int, default=10, help="Inclusive minimum hour to keep (default 10)")
    ap.add_argument("--hour-max", type=int, default=22, help="Inclusive maximum hour to keep (default 22)")
    ap.add_argument("--segment-radius", type=float, default=5.0, help="Meters to associate beacons with segments")
    ap.add_argument("--viz", type=str, choices=["beacons","segments","both"], default="beacons",
                    help="Visualization mode: beacons heatmap, segments intensity, or both")
    ap.add_argument("--limit-frames", type=int, default=None, help="Limit number of time frames (smoke test)")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    args = ap.parse_args()

    setup_logging(args.verbose)
    logging.info("Starting heatmap generation...")
    logging.info(f"Args: {args}")

    # Resolve defaults relative to script dir when needed
    def resolve_default(path):
        if os.path.exists(path):
            return path
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, path)
        return candidate if os.path.exists(candidate) else path

    args.beacons = resolve_default(args.beacons)
    args.segments = resolve_default(args.segments)
    args.counts = resolve_default(args.counts)

    logging.info(f"Using files:\n  beacons:  {args.beacons}\n  segments: {args.segments}\n  counts:   {args.counts}\n  output:   {args.out}")

    beacons, segments, counts = load_data(args.beacons, args.segments, args.counts, args.hour_min, args.hour_max)

    floors = sorted(beacons["floor"].unique())
    logging.info(f"Detected floors in beacons: {floors}")
    if args.floor.lower() == "all":
        targets = floors
    else:
        try:
            targets = [int(args.floor)]
        except Exception:
            raise ValueError("--floor must be 'all' or an integer.")
        for f in targets:
            if f not in floors:
                logging.warning(f"Requested floor {f} not in available floors {floors}")

    outputs = []
    for f in targets:
        t0 = time.time()
        logging.info(f"Processing floor {f} (viz={args.viz})...")
        out_path = args.out
        if len(targets) > 1:
            if "." in args.out:
                stem, ext = args.out.rsplit(".",1)
                out_path = f"{stem}_floor{f}.{ext}"
            else:
                out_path = f"{args.out}_floor{f}.gif"

        if args.viz == "beacons":
            frames_h = make_frames_heatmap(beacons, counts, floor=f, bins=args.bins,
                                           smooth_kernel=args.smooth_kernel, smooth_passes=args.smooth_passes)
            if args.limit_frames:
                frames_h = frames_h[:args.limit_frames]
            out_file = render_animation_heatmap(frames_h, segments, floor=f, out_path=out_path, fps=args.fps)
        elif args.viz == "segments":
            frames_s = make_frames_segments(beacons, segments, counts, floor=f, radius=args.segment_radius)
            if args.limit_frames:
                frames_s = frames_s[:args.limit_frames]
            out_file = render_animation_segments(frames_s, out_path=out_path, fps=args.fps)
        else:  # both
            frames_h = make_frames_heatmap(beacons, counts, floor=f, bins=args.bins,
                                           smooth_kernel=args.smooth_kernel, smooth_passes=args.smooth_passes)
            frames_s = make_frames_segments(beacons, segments, counts, floor=f, radius=args.segment_radius)
            if args.limit_frames:
                m = args.limit_frames
                frames_h = frames_h[:m]
                frames_s = frames_s[:m]
            out_file = render_animation_both(frames_h, frames_s, segments, floor=f, out_path=out_path, fps=args.fps)

        logging.info(f"Done floor {f} in {time.time()-t0:.2f}s -> {out_file}")
        outputs.append(out_file)

    logging.info("All done.")
    for p in outputs:
        print(f"[OK] Saved: {p}")

if __name__ == "__main__":
    main()