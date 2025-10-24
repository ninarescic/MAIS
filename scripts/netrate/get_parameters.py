import pandas as pd
import numpy as np
import networkx as nx
from numpy.linalg import eigvals

# Inputs
pred_path = "data/output/model/netrate_result_true.csv"         # source,target,beta (directed)
cascades_path = ("data/output/model/netrate/netrate_cascade_MODEL_beta=0.2_MODEL_I_duration=7_seed=1.csv")             # cascade_id,node_id,infection_time

# 1) β_global from NetRate
pred = pd.read_csv(pred_path)
pred = pred[pred["source"] != pred["target"]].copy()
beta_global = pred["beta"].mean()   # or .median() for robustness

# 2) ρ(A): spectral radius of directed adjacency on inferred edges
G = nx.DiGraph()
G.add_weighted_edges_from(pred[["source","target","beta"]].itertuples(index=False, name=None))
A = nx.to_numpy_array(G, nodelist=list(G.nodes()), weight=None)  # unweighted adjacency for threshold dynamics
rho = max(abs(eigvals(A))) if A.size else 0.0

# 3) r: early exponential growth from cascades (pooled)
cas = pd.read_csv(cascades_path)
# choose a time origin per cascade and pool; bin by small dt
dt = 1.0  # tune to your time units; try 0.5 or 1.0
cas["infection_time"] = cas["infection_time"].astype(float)

# Build pooled incidence over time (relative to each cascade's first infection)
inc_rows = []
for cid, dfc in cas.groupby("cascade_id"):
    t0 = dfc["infection_time"].min()
    t_rel = dfc["infection_time"] - t0
    bins = np.floor(t_rel/dt).astype(int)
    inc = pd.Series(1, index=bins).groupby(level=0).sum().sort_index()
    for k, c in inc.items():
        inc_rows.append({"bin": k, "count": c})
inc = pd.DataFrame(inc_rows).groupby("bin")["count"].sum().sort_index()

# Keep only the early phase (first 5–8 bins with >0); linear fit on log
inc = inc[inc > 0]
early = inc.iloc[:min(8, len(inc))]
if len(early) >= 3:
    x = early.index.values.astype(float) * dt
    y = np.log(early.values.astype(float))
    # ordinary least squares slope = r
    r = np.polyfit(x, y, 1)[0]
else:
    r = 0.0  # fallback if too little data

# 4) γ from r ≈ β_global * ρ(A) − γ
gamma = float(beta_global * rho - r)
gamma = max(gamma, 0.0)

print(f"β_global (from NetRate mean): {beta_global:.6f}")
print(f"ρ(A) (spectral radius):       {rho:.6f}")
print(f"Early growth rate r:           {r:.6f}")
print(f"γ (recovery rate):             {gamma:.6f}")
