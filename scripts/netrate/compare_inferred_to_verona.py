#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare NetRate inferred edges against Verona (or any) ground truth.

Supports:
- Directed truth (exact edge (u->v) match)
- Undirected truth (pair {u,v} match), with "pairwise-collapsed" predicted ranking (max beta across directions)
- Directional diagnostics when truth is undirected

Inputs:
  Predicted: CSV with columns [source,target,beta]
  Truth: CSV with 2 columns (by default: vertex1,vertex2). Use --u_col/--v_col to override.

Examples
--------
1) Directed truth (recommended if your Verona is directional):
   python scripts/netrate/compare_inferred_to_verona.py \
     --pred data/output/model/netrate_result_true.csv \
     --truth data/m-input/verona/raj-full-edges.csv \
     --directed_truth 1 --k 500

2) Undirected truth + pairwise collapsed evaluation (historical compatibility),
   plus a directional diagnostic:
   python scripts/netrate/compare_inferred_to_verona.py \
     --pred data/output/model/netrate_result_true.csv \
     --truth data/m-input/verona/raj-full-edges.csv \
     --directed_truth 0 --k 500

"""

import argparse
from pathlib import Path
import pandas as pd

def load_pred(path, thr=0.0, topk=0):
    df = pd.read_csv(path)
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("Prediction CSV must contain columns: source,target[,beta].")
    if "beta" not in df.columns:
        df["beta"] = 1.0
    # normalize and de-duplicate by keeping strongest beta per directed edge
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df = df[df["source"] != df["target"]].copy()
    df = df.sort_values("beta", ascending=False)
    df = df.groupby(["source", "target"], as_index=False)["beta"].max()
    if thr > 0:
        df = df[df["beta"] > thr]
    df = df.sort_values("beta", ascending=False).reset_index(drop=True)
    if topk and topk > 0 and len(df) > topk:
        df = df.head(topk).copy()
    return df

def load_truth(path, u_col=None, v_col=None, directed_truth=False):
    tdf = pd.read_csv(path, dtype=str)
    if u_col is None or v_col is None:
        # try common names, else first two columns
        if {"vertex1", "vertex2"}.issubset(tdf.columns):
            u_col, v_col = "vertex1", "vertex2"
        else:
            cols = list(tdf.columns)
            u_col, v_col = cols[0], cols[1]
    tdf = tdf[[u_col, v_col]].copy()
    tdf[u_col] = tdf[u_col].astype(str)
    tdf[v_col] = tdf[v_col].astype(str)
    if directed_truth:
        truth_dir = set((u, v) for u, v in tdf.itertuples(index=False, name=None) if u != v)
        return truth_dir, None
    else:
        truth_und = set(frozenset((u, v)) for u, v in tdf.itertuples(index=False, name=None) if u != v)
        return None, truth_und

def prf_counts(pred_set, truth_set):
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = (2 * prec * rec) / max(1e-12, (prec + rec))
    return tp, fp, fn, prec, rec, f1

def collapse_to_pairs(pred_df):
    """
    Collapse directed predictions to undirected pairs by taking max beta across (u,v) and (v,u).
    Returns a DataFrame with columns: pair_u, pair_v, beta_pair (u < v lexicographically for stable keys).
    """
    rows = {}
    for _, r in pred_df.iterrows():
        u, v, b = r["source"], r["target"], float(r["beta"])
        if u == v:
            continue
        a, c = (u, v) if u < v else (v, u)
        key = (a, c)
        if key not in rows or b > rows[key]:
            rows[key] = b
    out = pd.DataFrame(
        [(a, c, rows[(a, c)]) for (a, c) in rows.keys()],
        columns=["pair_u", "pair_v", "beta_pair"]
    )
    out = out.sort_values("beta_pair", ascending=False).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Compare NetRate predictions to truth (directed or undirected).")
    ap.add_argument("--pred", required=True, help="Predicted CSV with columns source,target,beta")
    ap.add_argument("--truth", required=True, help="Truth CSV with two columns (u,v)")
    ap.add_argument("--u_col", default=None, help="Column name in truth for source/left node")
    ap.add_argument("--v_col", default=None, help="Column name in truth for target/right node")
    ap.add_argument("--directed_truth", type=int, default=0, help="1 if truth is directed; 0 if undirected")
    ap.add_argument("--k", type=int, default=0, help="Evaluate on top-K predictions (0 = use all after threshold)")
    ap.add_argument("--thr", type=float, default=0.0, help="Drop predicted edges with beta <= thr before ranking")
    ap.add_argument("--also_write_matches", default="", help="Optional path to write a CSV of matches/mismatches")
    args = ap.parse_args()

    pred_df = load_pred(args.pred, thr=args.thr, topk=args.k)
    truth_dir, truth_und = load_truth(args.truth, args.u_col, args.v_col, directed_truth=bool(args.directed_truth))

    print(f"Loaded predictions: {len(pred_df)} edges after threshold/topK.")
    if truth_dir is not None:
        print(f"Loaded DIRECTED truth edges: {len(truth_dir)}")
    else:
        print(f"Loaded UNDIRECTED truth pairs: {len(truth_und)}")

    # --- Evaluation A: Directed (only if truth is directed) ---
    if truth_dir is not None:
        pred_dir_set = set((u, v) for u, v in pred_df[["source", "target"]].itertuples(index=False, name=None))
        tp, fp, fn, prec, rec, f1 = prf_counts(pred_dir_set, truth_dir)
        print("\n=== Directed evaluation (exact (u->v) match) ===")
        print(f"TP={tp}  FP={fp}  FN={fn}")
        print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    # --- Evaluation B: Undirected pairs (pairwise-collapsed) ---
    # Useful for historical comparisons or when truth is undirected
    collapsed = collapse_to_pairs(pred_df)  # ranked by beta_pair
    if args.k and args.k > 0 and len(collapsed) > args.k:
        collapsed_eval = collapsed.head(args.k)
    else:
        collapsed_eval = collapsed

    pred_pairs_set = set(frozenset((u, v)) for u, v in collapsed_eval[["pair_u", "pair_v"]].itertuples(index=False, name=None))

    # If truth is directed, derive an undirected set from it for pairwise evaluation
    if truth_dir is not None:
        truth_pairs = set(frozenset((u, v)) for (u, v) in truth_dir)
    else:
        truth_pairs = truth_und

    tp_p, fp_p, fn_p, prec_p, rec_p, f1_p = prf_counts(pred_pairs_set, truth_pairs)
    print("\n=== Undirected pairwise evaluation (collapse directions by max beta) ===")
    print(f"TP={tp_p}  FP={fp_p}  FN={fn_p}")
    print(f"Precision={prec_p:.4f}  Recall={rec_p:.4f}  F1={f1_p:.4f}")

    # --- Evaluation C: Directional diagnostic (only meaningful if truth is undirected) ---
    if truth_dir is None:
        # Among predicted directed edges that land on a true undirected pair,
        # how often did we predict both directions vs only one?
        on_true = pred_df[pred_df.apply(lambda r: frozenset((r["source"], r["target"])) in truth_pairs, axis=1)]
        dir_pairs = {}
        for _, r in on_true.iterrows():
            u, v = r["source"], r["target"]
            key = tuple(sorted((u, v)))
            dir_pairs.setdefault(key, set()).add((u, v))
        both_dirs = sum(1 for s in dir_pairs.values() if len(s) == 2)
        one_dir  = sum(1 for s in dir_pairs.values() if len(s) == 1)
        print("\n=== Directional diagnostic (truth is UNDIRECTED) ===")
        print(f"Predicted true pairs with both directions: {both_dirs}")
        print(f"Predicted true pairs with one direction:  {one_dir}")

    # Optional: write a row-wise file of matches/misses
    if args.also_write_matches:
        out_rows = []
        truth_dir_set = truth_dir if truth_dir is not None else set()
        truth_pair_set = truth_pairs

        for _, r in pred_df.iterrows():
            u, v, b = r["source"], r["target"], float(r["beta"])
            dir_hit = int((u, v) in truth_dir_set) if truth_dir is not None else None
            und_hit = int(frozenset((u, v)) in truth_pair_set)
            out_rows.append({"source": u, "target": v, "beta": b,
                             "hit_directed": dir_hit, "hit_undirected": und_hit})

        out_df = pd.DataFrame(out_rows)
        out_path = Path(args.also_write_matches)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nWrote detailed matches to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
