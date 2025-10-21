from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
res_path = ROOT / "data" / "output" / "model" / "netrate_result_true.csv"   # change if needed
edges_path = ROOT / "data" / "m-input" / "verona" / "raj-full-edges.csv"

# Load inferred edges
inf = pd.read_csv(res_path)
inf["source"] = inf["source"].astype(str).str.strip()
inf["target"] = inf["target"].astype(str).str.strip()

# Load ground-truth edges with explicit endpoint columns
gt = pd.read_csv(edges_path, dtype=str)
cols = [c.lower() for c in gt.columns]
if "vertex1" in gt.columns and "vertex2" in gt.columns:
    u_col, v_col = "vertex1", "vertex2"
else:
    # fallback: try common names, else first two (but warn)
    candidates = [("u","v"), ("src","dst"), ("source","target"), ("from","to")]
    for a,b in candidates:
        if a in cols and b in cols:
            u_col = gt.columns[cols.index(a)]
            v_col = gt.columns[cols.index(b)]
            break
    else:
        # LAST resort: first two columns (but this is likely wrong for Verona)
        u_col, v_col = gt.columns[:2]
        print(f"[WARN] Falling back to first two columns: ({u_col},{v_col}). For Verona you should use (vertex1,vertex2).")

gt["u"] = gt[u_col].astype(str).str.strip()
gt["v"] = gt[v_col].astype(str).str.strip()

# Build undirected set of true edges
true_und = {frozenset((u, v)) for u, v in zip(gt["u"], gt["v"]) if u != v}

# Evaluate at top-K by beta, where K = |true edges|
true_K = len(true_und)
inf_sorted = inf.sort_values("beta", ascending=False).reset_index(drop=True)

picked = []
seen_und = set()
for _, row in inf_sorted.iterrows():
    e = frozenset((str(row["source"]), str(row["target"])))
    if len(e) == 1:  # skip self-loops
        continue
    if e in seen_und:
        continue
    seen_und.add(e)
    picked.append((row["source"], row["target"], row["beta"]))
    if len(seen_und) == true_K:
        break

pred_und = set(seen_und)
tp = len(pred_und & true_und)
precision_at_k = tp / true_K
recall_at_k = tp / true_K

print(f"True undirected edges: {len(true_und)}")
print(f"Top-K predicted (K=true edges): {len(pred_und)}")
print(f"TP overlap: {tp}")
print(f"Precision@K: {precision_at_k:.3f}  Recall@K: {recall_at_k:.3f}")

def undirected_hit(row):
    return frozenset((str(row['source']), str(row['target']))) in true_und

print("\nTop 15 preview (beta & hit):")
prev = inf_sorted.head(15).copy()
prev["in_ground_truth"] = prev.apply(undirected_hit, axis=1)
print(prev[["source","target","beta","in_ground_truth"]].to_string(index=False))
