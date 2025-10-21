# scripts/netrate/visualize_netrate_vs_true.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("TkAgg")  # or comment out if you prefer the default backend
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from scripts.netrate.utils_netrate import project_root
from pathlib import Path


# -------- paths (root is MAIS/) ----------
ROOT = project_root()
TRUE_EDGES = ROOT / "data" / "m-input" / "verona" / "raj-full-edges.csv"
INFERRED = ROOT / "data" / "output" / "model" / "netrate_result_true.csv"  # change if you use a different filename

# -------- load true graph ----------
true_df = pd.read_csv(TRUE_EDGES, dtype=str)
if {"vertex1", "vertex2"}.issubset(true_df.columns):
    u_col, v_col = "vertex1", "vertex2"
else:
    # fallback: first two columns
    u_col, v_col = true_df.columns[:2]
true_edges = list(zip(true_df[u_col], true_df[v_col]))
G_true = nx.Graph()
G_true.add_edges_from(true_edges)

# -------- load inferred graph ----------
inf_df = pd.read_csv(INFERRED)
inf_df["source"] = inf_df["source"].astype(str)
inf_df["target"] = inf_df["target"].astype(str)

# keep exactly as many strongest edges as in the true graph (undirected count)
K = G_true.number_of_edges()
inf_df = inf_df.sort_values("beta", ascending=False)
picked = []
seen = set()
for _, r in inf_df.iterrows():
    e = frozenset((r["source"], r["target"]))
    if len(e) == 1 or e in seen:
        continue
    seen.add(e)
    picked.append((r["source"], r["target"]))
    if len(seen) >= K:
        break

G_inf = nx.Graph()
G_inf.add_edges_from(picked)

# -------- build overlay graph with colors ----------
G_all = nx.Graph()
G_all.add_edges_from(G_true.edges())
G_all.add_edges_from(G_inf.edges())

true_und = {frozenset(e) for e in G_true.edges()}
inf_und  = {frozenset(e) for e in G_inf.edges()}

edge_colors = []
for u, v in G_all.edges():
    e = frozenset((u, v))
    if e in true_und and e in inf_und:
        edge_colors.append("green")       # hit
    elif e in inf_und and e not in true_und:
        edge_colors.append("red")         # false positive
    else:
        edge_colors.append("lightgray")   # missed true edge

# -------- draw & save ----------
pos = nx.spring_layout(G_all, seed=42)
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G_all, pos, node_size=140, node_color="skyblue", edgecolors="k", linewidths=0.5)
nx.draw_networkx_edges(G_all, pos, edge_color=edge_colors, width=1.5, alpha=0.85)
plt.title("NetRate vs True Verona\nGreen: correct  Red: false positive  Gray: missed")
plt.axis("off")

out = ROOT / "data" / "output" / "model" / "verona_netrate_overlay.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved figure to: {out}")
# plt.show()  # enable if you want an interactive window
