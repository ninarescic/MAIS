# scripts/outbreaker2/compare_outbreaker_to_verona.py

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Compare Python-outbreaker links vs Verona ground truth (undirected, Top-K)")
    ap.add_argument("--pred", required=True, help="CSV with columns: source,target,support")
    ap.add_argument("--verona", required=True, help="Ground truth edges CSV (vertex1,vertex2)")
    ap.add_argument("--topk_mode", choices=["K_true", "all"], default="K_true",
                    help="If K_true, take Top-K = true edge count; if all, use all predictions.")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    verona_path = Path(args.verona)
    if not pred_path.exists():
        raise SystemExit(f"Pred file not found: {pred_path}")
    if not verona_path.exists():
        raise SystemExit(f"Verona file not found: {verona_path}")

    pred = pd.read_csv(pred_path)
    pred["source"] = pred["source"].astype(str).str.strip()
    pred["target"] = pred["target"].astype(str).str.strip()
    pred = pred.sort_values("support", ascending=False).reset_index(drop=True)

    gt = pd.read_csv(verona_path, dtype=str)
    if {"vertex1", "vertex2"}.issubset(gt.columns):
        u_col, v_col = "vertex1", "vertex2"
    else:
        u_col, v_col = gt.columns[:2]

    true_und = {frozenset((u, v)) for u, v in zip(gt[u_col], gt[v_col]) if u != v}
    K = len(true_und)

    # unique undirected predicted edges
    seen, picked = set(), []
    for _, r in pred.iterrows():
        e = frozenset((r["source"], r["target"]))
        if len(e) == 1 or e in seen:
            continue
        seen.add(e)
        picked.append((r["source"], r["target"], r["support"]))
        if args.topk_mode == "K_true" and len(seen) >= K:
            break

    pred_und = set(seen)
    tp = len(pred_und & true_und)
    denom = K if args.topk_mode == "K_true" else max(1, len(pred_und))
    precision_at_k = tp / denom
    recall_at_k = tp / K if K else 0.0

    print(f"True undirected edges: {K}")
    print(f"Top-K predicted (K=true edges): {len(pred_und)}")
    print(f"TP overlap: {tp}")
    print(f"Precision@K: {precision_at_k:.3f}  Recall@K: {recall_at_k:.3f}")

    # preview top 15
    prev = pd.DataFrame(picked, columns=["source", "target", "support"]).head(15)
    prev["in_ground_truth"] = prev.apply(lambda r: frozenset((r["source"], r["target"])) in true_und, axis=1)
    print("\nTop 15 preview (support & hit):")
    print(prev.to_string(index=False))


if __name__ == "__main__":
    main()
