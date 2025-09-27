#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de flux indoor (beacons / segments) – sorties tabulaires + rapport HTML.

Entrées (CSV) attendues (en minuscules après normalisation des entêtes) :
- beacons:  id, x_m, y_m, floor
- segments: segment_id, x1_m, y1_m, x2_m, y2_m, floor
            (optionnel) type  -> si présent, on peut filtrer ascenseurs (ex: 'elevator'), garder escalators/escaliers
- counts:   floor, beacon, date_it, hour, count, direction_out, direction_in

Sorties :
- CSV: segment_loads.csv, top_segments.csv, top_od_beacons.csv, inter_floor_matrix.csv
- HTML: report_paths.html (bar charts, heatmap, sankey)
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# -------- utils --------
def setup_logging(v: int):
    level = logging.WARNING
    if v == 1: level = logging.INFO
    if v >= 2: level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

def read_csv_lower(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_cols(df: pd.DataFrame, need, name="df"):
    if any(n in (df.index.names or []) for n in need):
        df = df.reset_index()
    # rattrapage floor_x / floor_y
    for n in need:
        if n not in df.columns:
            if f"{n}_x" in df.columns: df = df.rename(columns={f"{n}_x": n})
            if f"{n}_y" in df.columns and n not in df.columns: df = df.rename(columns={f"{n}_y": n})
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: colonnes manquantes {missing}. Colonnes dispo: {list(df.columns)}")
    return df

def point_to_segment_distance(px, py, x1, y1, x2, y2) -> float:
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx*wx + vy*wy
    if c1 <= 0: return float(np.hypot(px - x1, py - y1))
    c2 = vx*vx + vy*vy
    if c2 <= c1: return float(np.hypot(px - x2, py - y2))
    b = c1 / c2
    bx, by = x1 + b*vx, y1 + b*vy
    return float(np.hypot(px - bx, py - by))

# -------- core --------
def load_data(beacons_csv, segments_csv, counts_csv, hour_min, hour_max,
              date_start: Optional[str], date_end: Optional[str],
              dir_map_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b = read_csv_lower(beacons_csv)
    s = read_csv_lower(segments_csv)
    c = read_csv_lower(counts_csv)

    need_b = ["id","x_m","y_m","floor"]
    need_s = ["segment_id","x1_m","y1_m","x2_m","y2_m","floor"]
    need_c = ["floor","beacon","date_it","hour","count"]
    b = ensure_cols(b, need_b, "beacons")
    s = ensure_cols(s, need_s, "segments")
    c = ensure_cols(c, need_c, "counts")

    b = to_num(b, ["x_m","y_m","floor"]).dropna(subset=["id","x_m","y_m","floor"])
    s = to_num(s, ["x1_m","y1_m","x2_m","y2_m","floor"]).dropna(subset=need_s)
    c = to_num(c, ["floor","hour","count"]).dropna(subset=["beacon","floor","date_it","hour","count"])

    b["floor"] = b["floor"].astype(int)
    s["floor"] = s["floor"].astype(int)
    c["floor"] = c["floor"].astype(int)
    c["hour"]  = c["hour"].astype(int)
    c["count"] = c["count"].astype(float)

    # filtre heure
    pre = len(c)
    c = c[(c["hour"] >= hour_min) & (c["hour"] <= hour_max)].copy()
    logging.info(f"Hour filter [{hour_min},{hour_max}] kept {len(c):,}/{pre:,}")

    # timestamp
    c["ts"] = pd.to_datetime(c["date_it"], errors="coerce") + pd.to_timedelta(c["hour"], unit="h")
    c = c.dropna(subset=["ts"]).copy()

    # filtre date (fin inclusive)
    if date_start or date_end:
        ds = pd.to_datetime(date_start) if date_start else c["ts"].min()
        de = pd.to_datetime(date_end)   if date_end   else c["ts"].max()
        de_excl = de + pd.Timedelta(days=1)
        pre = len(c)
        c = c[(c["ts"] >= ds) & (c["ts"] < de_excl)]
        logging.info(f"Date filter [{ds}..{de}] inclusive kept {len(c):,}/{pre:,}")
    if c.empty:
        raise RuntimeError("No data after hour/date filters.")

    # garantir présence des colonnes directions
    if "direction_out" not in c.columns: c["direction_out"] = np.nan
    if "direction_in"  not in c.columns: c["direction_in"]  = np.nan

    # ---- MAPPING ICI (bon endroit) ----
    # Normalisation helper
    def _norm(sr: pd.Series) -> pd.Series:
        return (sr.astype(str).str.strip().str.upper().str.replace(r"\s+","",regex=True))

    # Appliquer un CSV de mapping optionnel: direction_code -> beacon_id
    if dir_map_path:
        dm = pd.read_csv(dir_map_path, dtype=str)
        dm.columns = [c.strip().lower() for c in dm.columns]
        if not {"direction_code","beacon_id"} <= set(dm.columns):
            raise ValueError("--dir-map doit contenir les colonnes: direction_code, beacon_id")
        dm["direction_code_norm"] = _norm(dm["direction_code"])
        dm["beacon_id_norm"]      = _norm(dm["beacon_id"])
        map_series = dm.set_index("direction_code_norm")["beacon_id_norm"]
        # map -> si code inconnu, on garde la valeur normalisée (peut déjà être un id)
        c["direction_out"] = _norm(c["direction_out"]).map(map_series).fillna(_norm(c["direction_out"]))
        c["direction_in"]  = _norm(c["direction_in"]).map(map_series).fillna(_norm(c["direction_in"]))
    else:
        # pas de mapping fourni : on normalise quand même (peut suffire si directions == ids)
        c["direction_out"] = _norm(c["direction_out"])
        c["direction_in"]  = _norm(c["direction_in"])
    # -----------------------------------

    return b, s, c

def build_beacon_flows(b: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des 'flux' à partir des directions de type {-1,0,1,outdoor}.
    Ici, src = floor du beacon (où est mesuré le passage), dst = direction_out (bucket).
    On n'essaie PAS de joindre à un beacon destination.
    """
    # normalisation légère des directions
    def norm_dir(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip().str.lower()
        # uniformiser quelques variantes éventuelles
        s = s.replace({"-1":"-1","0":"0","1":"1","out":"outdoor","exterior":"outdoor","outdoor":"outdoor"})
        return s

    # beacons -> positions (pour projeter sur segments de l'étage)
    bf = b[["id","x_m","y_m","floor"]].rename(columns={"id":"beacon"})
    cl = c.copy()
    cl["direction_out"] = norm_dir(cl.get("direction_out", pd.Series(index=cl.index, dtype=str)))
    cl["direction_in"]  = norm_dir(cl.get("direction_in",  pd.Series(index=cl.index, dtype=str)))

    # on ne garde que les directions connues
    valid = set(["-1","0","1","outdoor"])
    cl = cl[cl["direction_out"].isin(valid) | cl["direction_in"].isin(valid)].copy()
    if cl.empty:
        raise RuntimeError("Aucune ligne avec direction_in/out dans {-1,0,1,outdoor}.")

    # jointure pour récupérer coord/floor du beacon (lieu de comptage)
    e = cl.merge(bf, on=["beacon","floor"], how="inner", validate="many_to_one")
    if e.empty:
        raise RuntimeError("Impossible de joindre counts->beacons (vérifie la casse/espaces de 'beacon').")

    # src/dst au sens 'étage → bucket destination'
    e["src_floor"] = e["floor"].astype(int)
    # priorité à direction_out si dispo, sinon direction_in (rare, mais on le capte)
    e["dst_bucket"] = np.where(e["direction_out"].isin(valid), e["direction_out"], e["direction_in"])

    # poids
    e["weight"] = pd.to_numeric(e["count"], errors="coerce").fillna(0.0)

    # colonnes utiles pour la suite (projection segments)
    flows = e[["ts","beacon","x_m","y_m","src_floor","dst_bucket","weight"]].copy()
    return flows


def project_flows_to_segments(segments: pd.DataFrame, flows: pd.DataFrame, radius: float = 5.0) -> pd.DataFrame:
    """
    Projette des flux sur les segments.
    Deux modes supportés automatiquement :
      A) Flux ligne src→dst (colonnes présentes: x_m,y_m,x_m_dst,y_m_dst) :
         on approxime la distance ligne-segment par min(distance des deux extrémités au segment).
      B) Flux "bucket" sans coordonnées dst (colonnes: x_m,y_m,src_floor[,dst_bucket]) :
         on projette le point (beacon) sur les segments du MÊME ÉTAGE (src_floor).
    Retourne un DataFrame segments + 'flow_load' (agrégé sur toute la période).
    """
    seg = segments.copy().reset_index(drop=True)
    seg["idx"] = seg.index

    # Arrays pour perf
    sx1 = seg["x1_m"].to_numpy(float); sy1 = seg["y1_m"].to_numpy(float)
    sx2 = seg["x2_m"].to_numpy(float); sy2 = seg["y2_m"].to_numpy(float)

    vals = np.zeros(len(seg), dtype=float)

    has_dst_coords = {"x_m_dst","y_m_dst"}.issubset(flows.columns)
    has_bucket_mode = {"x_m","y_m","src_floor"}.issubset(flows.columns)

    if has_dst_coords:
        # MODE A : ligne src→dst
        for _, r in flows.iterrows():
            fx1, fy1 = float(r["x_m"]),     float(r["y_m"])
            fx2, fy2 = float(r["x_m_dst"]), float(r["y_m_dst"])
            w = float(r["weight"])

            # distance aux segments : min(dist(src,segment), dist(dst,segment))
            dists = np.array([min(
                point_to_segment_distance(fx1,fy1,x1,y1,x2,y2),
                point_to_segment_distance(fx2,fy2,x1,y1,x2,y2)
            ) for x1,y1,x2,y2 in zip(sx1,sy1,sx2,sy2)], dtype=float)

            mask = dists <= radius
            if not mask.any():
                continue
            wts = 1.0 / (1.0 + dists[mask])
            wts = wts / wts.sum()
            vals[mask] += w * wts

    elif has_bucket_mode:
        # MODE B : point → segments du même étage
        for f, g in flows.groupby("src_floor"):
            seg_f = seg[seg["floor"] == f]
            if seg_f.empty:
                continue
            idxs = seg_f["idx"].to_numpy()
            x1, y1, x2, y2 = sx1[idxs], sy1[idxs], sx2[idxs], sy2[idxs]

            for _, r in g.iterrows():
                bx, by, w = float(r["x_m"]), float(r["y_m"]), float(r["weight"])
                dists = np.array([point_to_segment_distance(bx,by, xi1,yi1, xi2,yi2)
                                  for xi1,yi1,xi2,yi2 in zip(x1,y1,x2,y2)], dtype=float)
                mask = dists <= radius
                if not mask.any():
                    continue
                wts = 1.0 / (1.0 + dists[mask])
                wts = wts / wts.sum()
                vals[idxs[mask]] += w * wts
    else:
        raise ValueError("Le DataFrame 'flows' ne contient ni (x_m_dst,y_m_dst) ni (src_floor). Vérifie la construction des flux.")

    seg["flow_load"] = vals
    return seg

def build_inter_floor_matrix_from_buckets(flows: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la matrice 'from_floor -> to_bucket' sur {-1,0,1,outdoor}.
    """
    m = (flows.groupby(["src_floor","dst_bucket"], as_index=False)["weight"].sum()
              .pivot(index="src_floor", columns="dst_bucket", values="weight")
              .reindex(index=[-1,0,1], columns=["-1","0","1","outdoor"], fill_value=0.0))
    m.index.name = "from_floor"; m.columns.name = "to_bucket"
    return m.reset_index()

def project_counts_to_segments_by_bucket(segments: pd.DataFrame, flows: pd.DataFrame, radius: float = 5.0) -> dict:
    """
    Pour chaque bucket de destination ('-1','0','1','outdoor'), on projette les weights
    des beacons (x_m,y_m) du floor 'src_floor' vers les segments du MÊME étage (src_floor),
    en répartissant par 1/(1+d) sur les segments à distance <= radius.
    Retourne un dict bucket -> DataFrame seg_loads (avec 'flow_load').
    """
    buckets = ["-1","0","1","outdoor"]
    out = {}

    # préparer segments en arrays pour perf
    seg = segments.copy().reset_index(drop=True)
    seg["idx"] = seg.index
    sx1 = seg["x1_m"].to_numpy(float); sy1 = seg["y1_m"].to_numpy(float)
    sx2 = seg["x2_m"].to_numpy(float); sy2 = seg["y2_m"].to_numpy(float)

    for buck in buckets:
        sub = flows[flows["dst_bucket"] == buck]
        if sub.empty:
            # retourne un DF vide mais aux mêmes colonnes pour homogénéité
            out[buck] = seg.assign(flow_load=0.0).copy()
            continue

        # on travaille étage par étage (projection locale)
        vals = np.zeros(len(seg), dtype=float)
        for f, g in sub.groupby("src_floor"):
            seg_f = seg[seg["floor"] == f]
            if seg_f.empty:
                continue
            idxs = seg_f["idx"].to_numpy()
            x1, y1, x2, y2 = sx1[idxs], sy1[idxs], sx2[idxs], sy2[idxs]

            for _, r in g.iterrows():
                bx, by, w = float(r["x_m"]), float(r["y_m"]), float(r["weight"])
                # distance beacon -> segment
                dists = np.array([point_to_segment_distance(bx, by, xi1, yi1, xi2, yi2) for xi1,yi1,xi2,yi2 in zip(x1,y1,x2,y2)], dtype=float)
                mask = dists <= radius
                if not mask.any():
                    continue
                wts = 1.0 / (1.0 + dists[mask])
                wts = wts / wts.sum()
                vals[idxs[mask]] += w * wts

        out[buck] = seg.assign(flow_load=vals).copy()

    return out

def top_k_segments(seg_loads: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    return seg_loads.sort_values("flow_load", ascending=False).head(k).copy()

def top_k_od_beacons(flows: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    od = (flows.groupby(["src","dst","floor_src","floor_dst"], as_index=False)["weight"].sum()
                .sort_values("weight", ascending=False)
                .head(k))
    return od

# -------- report (plotly) --------
def make_report_html(seg_loads: pd.DataFrame, topsegs: pd.DataFrame, topod: pd.DataFrame,
                     inter_floor: pd.DataFrame, out_html: str):
    import plotly.io as pio
    import plotly.express as px
    import plotly.graph_objects as go

    figs = []

    # Bar top segments
    if not topsegs.empty:
        f1 = px.bar(topsegs, x="segment_id", y="flow_load", title="Top segments (flow_load)",
                    labels={"segment_id":"segment","flow_load":"intensité"})
        f1.update_layout(xaxis_tickangle=-45)
        figs.append(f1)

    # Heatmap inter-floor (labels en str pour éviter le mix int/str)
    if not inter_floor.empty:
        M = inter_floor.copy()
        M["from_floor"] = M["from_floor"].astype(str)
        M = M.set_index("from_floor")

        # colonnes -> str
        M.columns = [str(c) for c in M.columns]

        # ordre conseillé
        desired_cols = ["-1", "0", "1", "outdoor"]
        cols_present = [c for c in desired_cols if c in M.columns]
        if cols_present:
            M = M.reindex(columns=cols_present)

        # ordre des lignes similaire
        desired_rows = ["-1", "0", "1"]
        rows_present = [r for r in desired_rows if r in M.index]
        if rows_present:
            M = M.reindex(index=rows_present)

        f2 = px.imshow(M, text_auto=True, title="Transitions entre étages (sum of weights)",
                       labels=dict(x="to_bucket", y="from_floor", color="flux"))
        figs.append(f2)

    # Sankey (agrégé par bucket) avec labels en str
    if not inter_floor.empty:
        MF = inter_floor.copy()
        MF["from_floor"] = MF["from_floor"].astype(str)
        # colonnes en str
        to_cols = [str(c) for c in MF.columns if c != "from_floor"]

        # ordre conseillé et filtrage aux colonnes présentes
        desired = ["-1", "0", "1", "outdoor"]
        labels = [lab for lab in desired if (lab in MF["from_floor"].unique()) or (lab in to_cols)]
        if not labels:
            labels = sorted(set(MF["from_floor"]).union(to_cols), key=str)

        idx = {lab: i for i, lab in enumerate(labels)}

        links = []
        for _, row in MF.iterrows():
            src = str(row["from_floor"])
            for col in to_cols:
                val = float(row[col])
                if val > 0 and (src in idx) and (col in idx):
                    links.append(dict(source=idx[src], target=idx[col], value=val))

        if links:
            f3 = go.Figure(data=[go.Sankey(
                node=dict(label=labels, pad=15, thickness=15),
                link=dict(
                    source=[l["source"] for l in links],
                    target=[l["target"] for l in links],
                    value=[l["value"] for l in links]
                )
            )])
            f3.update_layout(title_text="Sankey des flux par étage/bucket")
            figs.append(f3)

    # Table top OD beacons
    if not topod.empty:
        f4 = go.Figure(data=[go.Table(
            header=dict(values=["src","dst","floor_src","floor_dst","weight"], fill_color="lightgrey"),
            cells=dict(values=[topod[c] for c in ["src","dst","floor_src","floor_dst","weight"]])
        )])
        f4.update_layout(title="Top OD (beacons)")
        figs.append(f4)

    # Écriture HTML unique
    html_parts = []
    for i, fig in enumerate(figs, 1):
        html_parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False, auto_play=False))
    html = "<html><head><meta charset='utf-8'><title>Paths report</title></head><body>" + \
           "".join(html_parts) + \
           "</body></html>"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Analyse des patterns de cheminement (tableaux + rapport).")
    ap.add_argument("--beacons", type=str, required=True)
    ap.add_argument("--segments", type=str, required=True)
    ap.add_argument("--counts", type=str, required=True)
    ap.add_argument("--hour-min", type=int, default=10)
    ap.add_argument("--hour-max", type=int, default=22)
    ap.add_argument("--date-start", type=str, default=None)
    ap.add_argument("--date-end", type=str, default=None)
    ap.add_argument("--segment-radius", type=float, default=5.0, help="mètres de proximité pour projeter un flux sur des segments")
    ap.add_argument("--exclude-elevators", action="store_true", help="si colonne 'type' existe, on exclut type=='elevator'")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="analysis_out")
    ap.add_argument("-v","--verbose", action="count", default=0)
    args = ap.parse_args()

    setup_logging(args.verbose)
    os.makedirs(args.outdir, exist_ok=True)

    beacons, segments, counts = load_data(args.beacons, args.segments, args.counts, args.hour_min, args.hour_max, args.date_start, args.date_end)

    # option: exclure ascenseurs si 'type' présent
    if args.exclude_elevators and "type" in segments.columns:
        pre = len(segments)
        segments = segments[segments["type"].str.lower().ne("elevator")]
        logging.info(f"Segments: exclu 'elevator' ({pre-len(segments)}/{pre})")

    # 1) flux beacon→beacon
    flows = build_beacon_flows(beacons, counts)
    logging.info(f"Flows: {len(flows):,} arcs après jointure beacons.")

    # charges par segments, par destination (4 buckets)
    seg_loads_by_bucket = project_counts_to_segments_by_bucket(segments, flows, radius=args.segment_radius)

    # on peut aussi produire un 'overall' (somme des buckets) :
    seg_overall = segments.copy()
    seg_overall["flow_load"] = 0.0
    for dfb in seg_loads_by_bucket.values():
        seg_overall["flow_load"] += dfb["flow_load"]

    # 2) projection flux → segments (agrégée sur la période)
    seg_loads = project_flows_to_segments(segments, flows, radius=args.segment_radius)
    seg_loads = seg_loads.sort_values("flow_load", ascending=False)

    # 3) inter-floor matrix
    inter_floor = build_inter_floor_matrix_from_buckets(flows)

    # 4) Top k
    topsegs = top_k_segments(seg_loads, args.topk)
    has_beacon_od = {'src', 'dst', 'floor_src', 'floor_dst'}.issubset(flows.columns)
    topod = top_k_od_beacons(flows, args.topk) if has_beacon_od else pd.DataFrame()

    # 5) écritures CSV
    seg_loads.to_csv(os.path.join(args.outdir, "segment_loads.csv"), index=False)
    topsegs.to_csv(os.path.join(args.outdir, "top_segments.csv"), index=False)
    topod.to_csv(os.path.join(args.outdir, "top_od_beacons.csv"), index=False)
    inter_floor.to_csv(os.path.join(args.outdir, "inter_floor_matrix.csv"), index=False)
    seg_overall.to_csv(os.path.join(args.outdir, "segment_loads_overall.csv"), index=False)
    for buck, dfb in seg_loads_by_bucket.items():
        dfb.to_csv(os.path.join(args.outdir, f"segment_loads_bucket_{buck}.csv"), index=False)

    # 6) rapport HTML
    out_html = os.path.join(args.outdir, "report_paths.html")
    make_report_html(seg_loads, topsegs, topod, inter_floor, out_html)
    print(f"[OK] Wrote report: {out_html}")
    print(f"[OK] CSV: segment_loads.csv, top_segments.csv, top_od_beacons.csv, inter_floor_matrix.csv in {args.outdir}/")

if __name__ == "__main__":
    main()