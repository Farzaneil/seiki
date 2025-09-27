#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Interactive Evolving Heatmap (Plotly) with time slider and optional segments overlay.
#
# Features
# --------
# - Reads beacons, segments, counts CSVs (same schema as heatmap_evolutive.py)
# - Hour filtering (default 10-22)
# - Optional resampling (e.g., 30min)
# - Temporal smoothing (moving average on frames)
# - Normalization modes:
#     * global: fixed color scale across time
#     * per_frame: each frame scaled by its percentile range (relative)
#     * delta_baseline: H(t) - baseline (diverging colors)
#     * zscore_baseline: (H(t) - mu)/sigma (diverging, significance-like)
# - Output: single self-contained HTML with a slider + Play/Pause
# - Segments overlay drawn as static line shapes
#
# Usage
# -----
# python heatmap_interactive.py \
#   --beacons beacons_world_template.csv \
#   --segments segments_template.csv \
#   --counts porta_di_roma_counts.csv \
#   --floor 0 \
#   --out heatmap_interactive_floor0.html \
#   --bins 120 \
#   --hour-min 10 --hour-max 22 \
#   --time-granularity H \
#   --temporal-mode moving_avg --moving-window 3 \
#   --normalize per_frame --clip-percentile 99

import argparse
import logging
import os
import numpy as np
import pandas as pd
import plotly.io as pio

def ensure_cols(df, must_have, dfname="df"):
    # Ensure required names are present as COLUMNS (not index), robust to non-string column names.
    import logging

    index_names = list(df.index.names) if getattr(df.index, "names", None) is not None else []
    if any(name in index_names for name in must_have):
        df = df.reset_index()

    logging.debug(f"[ensure_cols] {dfname} initial cols: {list(df.columns)} | index: {index_names}")

    # Map from lowercase stringified column name -> actual column key
    lower_map = {}
    for c in df.columns:
        key = str(c).lower() if c is not None else ""
        lower_map[key] = c

    # Coerce needed names
    for needed in must_have:
        if needed not in df.columns:
            needed_l = needed.lower()
            if needed_l in lower_map and lower_map[needed_l] != needed:
                df = df.rename(columns={lower_map[needed_l]: needed})
            elif f"{needed}_x" in df.columns:
                df = df.rename(columns={f"{needed}_x": needed})
            elif f"{needed}_y" in df.columns:
                df = df.rename(columns={f"{needed}_y": needed})
            else:
                # fallback: normalized equality
                matches = [col for col in df.columns if str(col).lower() == needed_l]
                if matches:
                    m = matches[0]
                    if m != needed:
                        df = df.rename(columns={m: needed})

    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise ValueError(f"[ensure_cols] Missing in {dfname}: {missing}. Columns: {list(df.columns)} | index: {index_names}")
    return df


def setup_logging(verbosity: int):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

def load_data(beacons_csv: str, segments_csv: str, counts_csv: str, hour_min: int, hour_max: int, time_granularity: str, date_start=None, date_end=None):
    beacons = pd.read_csv(beacons_csv, dtype=str)
    segments = pd.read_csv(segments_csv, dtype=str)
    counts = pd.read_csv(counts_csv, dtype=str, low_memory=False)

    beacons.columns = [c.strip().lower() for c in beacons.columns]
    segments.columns = [c.strip().lower() for c in segments.columns]
    counts.columns = [c.strip().lower() for c in counts.columns]

    required_beacons = {"id","x_m","y_m","floor"}
    required_segments = {"segment_id","x1_m","x2_m","y1_m","y2_m","floor"}
    required_counts = {"floor","beacon","date_it","hour","count"}
    for req, df, name in [(required_beacons, beacons, "beacons"),
                          (required_segments, segments, "segments"),
                          (required_counts, counts, "counts")]:
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {name} CSV: {missing}")

    # numeric coercion
    for col in ["x_m","y_m","floor"]:
        beacons[col] = pd.to_numeric(beacons[col], errors="coerce")
    for col in ["x1_m","x2_m","y1_m","y2_m","floor"]:
        segments[col] = pd.to_numeric(segments[col], errors="coerce")
    for col in ["floor","hour","count"]:
        counts[col] = pd.to_numeric(counts[col], errors="coerce")

    beacons = beacons.dropna(subset=["id","x_m","y_m","floor"]).copy()
    segments = segments.dropna(subset=["segment_id","x1_m","x2_m","y1_m","y2_m","floor"]).copy()
    counts = counts.dropna(subset=["beacon","floor","date_it","hour","count"]).copy()

    beacons["floor"] = beacons["floor"].astype(int)
    segments["floor"] = segments["floor"].astype(int)
    counts["floor"] = counts["floor"].astype(int)
    counts["hour"] = counts["hour"].astype(int)
    counts["count"] = counts["count"].astype(float)

    # hour filter
    if hour_min is not None and hour_max is not None:
        before = len(counts)
        counts = counts[(counts["hour"] >= hour_min) & (counts["hour"] <= hour_max)]
        logging.info(f"Hour filter [{hour_min},{hour_max}] kept {len(counts):,}/{before:,} rows")

    # timestamp
    counts["ts"] = pd.to_datetime(counts["date_it"], errors="coerce") + pd.to_timedelta(counts["hour"], unit="h")
    counts = counts.dropna(subset=["ts"]).copy()

    # optional date range filter
    if date_start or date_end:
        ds = pd.to_datetime(date_start) if date_start else counts['ts'].min()
        de = pd.to_datetime(date_end) if date_end else counts['ts'].max()
        before = len(counts)
        counts = counts[(counts['ts'] >= ds) & (counts['ts'] <= de)]
        logging.info(f"Date filter [{ds}..{de}] kept {len(counts):,}/{before:,} rows")

    # resample if needed
    if time_granularity and time_granularity.upper() != "H":
        logging.info(f"Resampling to {time_granularity} by sum per beacon/floor")
        counts = (
            counts.set_index('ts')
                  .groupby(['floor','beacon'])
                  .resample(time_granularity)['count']
                  .sum()
                  .reset_index()
        )
        # ensure ts is datetime after reset_index
        counts['ts'] = pd.to_datetime(counts['ts'], errors='coerce')
        counts = counts.dropna(subset=['ts'])
        counts['date_it'] = counts['ts'].dt.date.astype(str)
        counts['hour'] = counts['ts'].dt.hour.astype(int)

    # Normalize columns to be sure after resampling
    counts = ensure_cols(counts, ["floor", "beacon", "ts", "count"], dfname="counts")

    return beacons, segments, counts

def make_frames_heatmap(beacons: pd.DataFrame, counts: pd.DataFrame, floor: int, bins: int = 120):
    counts = ensure_cols(counts, ["floor", "beacon", "ts", "count"], dfname="counts@make_frames")
    b_floor = beacons[beacons["floor"] == floor].copy()
    if b_floor.empty:
        raise ValueError(f"No beacons found for floor {floor}")
    c_floor = counts[counts["floor"] == floor].merge(
        b_floor.rename(columns={"id":"beacon"}), on=["beacon","floor"], how="inner"
    )
    if c_floor.empty:
        raise ValueError(f"No joined counts for floor {floor}")

    xmin, xmax = b_floor["x_m"].min(), b_floor["x_m"].max()
    ymin, ymax = b_floor["y_m"].min(), b_floor["y_m"].max()
    pad_x = max(1.0, 0.05*(xmax - xmin)); pad_y = max(1.0, 0.05*(ymax - ymin))
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    times = np.sort(c_floor["ts"].unique())
    frames = []
    for ts in times:
        df_t = c_floor[c_floor["ts"] == ts]
        H, xedges, yedges = np.histogram2d(
            df_t["x_m"].values.astype(float), df_t["y_m"].values.astype(float),
            bins=bins, range=[[xmin, xmax],[ymin, ymax]],
            weights=df_t["count"].values.astype(float),
        )
        H = H.T
        frames.append((ts, H, (xmin,xmax,ymin,ymax)))
    logging.info(f"Built {len(frames)} frames for floor {floor}")
    return frames

def rolling_frames(frames, window: int):
    if window <= 1:
        return frames
    out = []
    Hs = [H for _, H, _ in frames]
    ts = [ts for ts, _, _ in frames]
    for i in range(len(frames)):
        lo = max(0, i - window + 1)
        stack = np.stack(Hs[lo:i+1], axis=0)
        out.append((ts[i], np.nanmean(stack, axis=0), frames[i][2]))
    return out

def select_baseline(frames, start, end):
    bs = pd.to_datetime(start) if start else None
    be = pd.to_datetime(end) if end else None
    if bs is None and be is None:
        n = max(1, int(0.1 * len(frames)))
        stack = np.stack([H for _, H, _ in frames[:n]], axis=0)
    else:
        sel = []
        for ts, H, _ in frames:
            if (bs is None or ts >= bs) and (be is None or ts <= be):
                sel.append(H)
        if not sel:
            sel = [H for _, H, _ in frames]
        stack = np.stack(sel, axis=0)
    mu = np.nanmean(stack, axis=0)
    sd = np.nanstd(stack, axis=0) + 1e-6
    return mu, sd

def transform_frames(frames, normalize: str, clip_percentile: float, baseline_start=None, baseline_end=None):
    out = []
    if not frames:
        return frames, (0,1), "Viridis"
    if normalize == "per_frame":
        for ts, H, ext in frames:
            lo = float(np.nanpercentile(H, 100 - (100-clip_percentile)))
            hi = float(np.nanpercentile(H, clip_percentile))
            if hi <= lo:
                S = np.zeros_like(H)
            else:
                S = (H - lo) / (hi - lo)
                S = np.clip(S, 0, 1)
            out.append((ts, S, ext))
        return out, (0,1), "Viridis"
    elif normalize in ("delta_baseline","zscore_baseline"):
        mu, sd = select_baseline(frames, baseline_start, baseline_end)
        vmax = 0.0
        for ts, H, ext in frames:
            if normalize == "delta_baseline":
                T = H - mu
            else:
                T = (H - mu) / sd
            vmax = max(vmax, float(np.nanpercentile(np.abs(T), clip_percentile)))
            out.append((ts, T, ext))
        rng = (-vmax, vmax)
        return out, rng, "RdBu"
    else:
        vmax = max(float(np.nanpercentile(H, clip_percentile)) for _, H, _ in frames)
        for item in frames:
            out.append(item)
        return out, (0, vmax), "Viridis"

def build_plotly_html(frames, segments, beacons, floor, out_html: str, zrange, cmap_name: str, width: int = 900, height: int = 700, overlay_beacons: bool = False):
    import plotly.graph_objects as go
    ts0, H0, (xmin,xmax,ymin,ymax) = frames[0]
    x = np.linspace(xmin, xmax, H0.shape[1])
    y = np.linspace(ymin, ymax, H0.shape[0])

    is_blank = (np.all(~np.isfinite(H0)) or np.nanmax(np.abs(H0)) <= 1e-12)
    fig = go.Figure(
        data=[go.Heatmap(
            z=H0, x=x, y=y, zmin=zrange[0], zmax=zrange[1],
            colorscale=cmap_name, colorbar=dict(title="Intensity")
        )],
        layout=go.Layout(
            title=f"Floor {floor} — {pd.to_datetime(ts0).strftime('%Y-%m-%d %H:%M')}",
            xaxis=dict(title="x (m)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="y (m)"),
            width=width, height=height,
            shapes=[]
        )
    )

    seg_f = segments[segments["floor"] == floor]
    shapes = []
    for _, s in seg_f.iterrows():
        shapes.append(dict(
            type="line",
            x0=float(s["x1_m"]), y0=float(s["y1_m"]),
            x1=float(s["x2_m"]), y1=float(s["y2_m"]),
            line=dict(width=1, color="rgba(0,0,0,0.5)")
        ))
    fig.update_layout(shapes=shapes)
    if overlay_beacons:
        b_floor = beacons[beacons['floor'] == floor]
        if not b_floor.empty:
            fig.add_trace(go.Scattergl(x=b_floor['x_m'], y=b_floor['y_m'], mode='markers',
                                       marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='black')),
                                       name='Beacons'))
    if is_blank:
        fig.add_annotation(text='No visible signal for first frame (all zeros/NaN) — adjust filters or normalization',
                           xref='paper', yref='paper', x=0.5, y=0.95, showarrow=False, bgcolor='rgba(255,255,0,0.5)')

    plotly_frames = []
    slider_steps = []
    for i, (ts, H, _) in enumerate(frames):
        plotly_frames.append(go.Frame(
            data=[go.Heatmap(z=H, x=x, y=y, zmin=zrange[0], zmax=zrange[1], colorscale=cmap_name)],
            name=str(i),
            layout=go.Layout(title=f"Floor {floor} — {pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')}")
        ))
        slider_steps.append(dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')
        ))

    fig.update(frames=plotly_frames)

    sliders = [dict(
        active=0,
        pad={"t": 10},
        steps=slider_steps,
        x=0.05, y= -0.05, len=0.9
    )]

    fig.update_layout(
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.05, y=1.08,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(fromcurrent=True, frame=dict(duration=200, redraw=True), transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))]),
            ]
        )]
    )

    if True:
        import plotly.graph_objects as go
        probe = go.Figure(data=[go.Heatmap(z=[[0,1],[1,0]])])
        pio.write_html(probe, file=out_html.replace('.html', '_probe.html'), include_plotlyjs=True, full_html=True)
    pio.write_html(fig, file=out_html, include_plotlyjs=True, full_html=True)
    return out_html

def main():
    parser = argparse.ArgumentParser(description="Interactive heatmap HTML with time slider (Plotly)")
    parser.add_argument("--beacons", type=str, default="beacons_world_template.csv")
    parser.add_argument("--segments", type=str, default="segments_template.csv")
    parser.add_argument("--counts", type=str, default="porta_di_roma_counts.csv")
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--out", type=str, default="heatmap_interactive.html")
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--hour-min", type=int, default=10)
    parser.add_argument("--hour-max", type=int, default=22)
    parser.add_argument("--time-granularity", type=str, default="H")
    parser.add_argument("--temporal-mode", type=str, choices=["per_interval","moving_avg"], default="per_interval")
    parser.add_argument("--moving-window", type=int, default=3)
    parser.add_argument("--normalize", type=str, choices=["global","per_frame","delta_baseline","zscore_baseline"], default="global")
    parser.add_argument("--baseline-start", type=str, default=None)
    parser.add_argument("--baseline-end", type=str, default=None)
    parser.add_argument("--clip-percentile", type=float, default=99.0)
    parser.add_argument("--date-start", type=str, default=None)
    parser.add_argument("--date-end", type=str, default=None)
    parser.add_argument("--snapshot", type=str, default=None, help="Save first-frame PNG for quick check")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--overlay-beacons", action="store_true", help="Draw beacon points on top for visual anchor")
    parser.add_argument("--write-probe-html", action="store_true", help="Also emit a tiny Plotly probe HTML to verify rendering")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    setup_logging(args.verbose)
    logging.info(f"Args: {args}")

    def resolve_default(path):
        if os.path.exists(path):
            return path
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, path)
        return candidate if os.path.exists(candidate) else path

    bea = resolve_default(args.beacons)
    seg = resolve_default(args.segments)
    cou = resolve_default(args.counts)

    beacons, segments, counts = load_data(bea, seg, cou, args.hour_min, args.hour_max, args.time_granularity, args.date_start, args.date_end)
    frames = make_frames_heatmap(beacons, counts, floor=args.floor, bins=args.bins)

    if args.temporal_mode == "moving_avg" and args.moving_window > 1:
        frames = rolling_frames(frames, args.moving_window)

    frames_t, zrange, cmap = transform_frames(frames, args.normalize, args.clip_percentile, args.baseline_start, args.baseline_end)
    # Debug stats
    import numpy as np
    if not frames_t:
        raise RuntimeError('No frames to render after transforms. Check filters (hour/date) and joins.')
    H0 = frames_t[0][1]
    print('[DEBUG] First frame shape:', H0.shape, 'min=', np.nanmin(H0), 'max=', np.nanmax(H0), 'nan%=', np.isnan(H0).mean()*100)
    print('[DEBUG] zrange:', zrange, 'cmap:', cmap)
    # Optional snapshot
    if args.snapshot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(H0, origin='lower', aspect='equal')
            plt.colorbar(im, ax=ax)
            fig.savefig(args.snapshot, dpi=160, bbox_inches='tight')
            plt.close(fig)
            print('[OK] Saved snapshot to', args.snapshot)
        except Exception as e:
            print('[WARN] Snapshot failed:', e)

    out_path = build_plotly_html(frames_t, segments, beacons, args.floor, args.out, zrange, cmap_name=cmap, width=args.width, height=args.height, overlay_beacons=args.overlay_beacons)
    print(f"[OK] Wrote interactive HTML to: {out_path}")

if __name__ == "__main__":
    main()