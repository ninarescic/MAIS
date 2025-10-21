from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "output" / "model"
GRAPH_NODES = ROOT / "data" / "m-input" / "verona" / "raj-nodes.csv"  # from your ZIP header
OUT_CSV = OUT_DIR / "netrate_cascades.csv"

rng = np.random.default_rng(42)

# Load node list; assume a column named 'id' or similar
nodes_df = pd.read_csv(GRAPH_NODES)
node_col = next((c for c in nodes_df.columns if c.lower() in {"id","node","node_id"}), None)
if node_col is None:
    raise SystemExit(f"Cannot find node id column in {GRAPH_NODES}. Columns: {list(nodes_df.columns)}")
all_nodes = nodes_df[node_col].astype(str).tolist()

# Find your history zips
zips = sorted(OUT_DIR.glob("history_netrate_agent_test_MODEL_*.zip"))
if not zips:
    zips = sorted(OUT_DIR.glob("history_*.zip"))
if not zips:
    raise SystemExit("No history_*.zip files found.")

rows = []
cascade_id = 0

for z in zips:
    with zipfile.ZipFile(z) as zz:
        # take the first CSV inside (they look like *_0.csv)
        csv_names = [n for n in zz.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            continue
        with zz.open(csv_names[0]) as f:
            # Skip comment lines starting with '#'
            df = pd.read_csv(f, sep=None, engine="python", comment="#")
    # Expect columns: inc_I and either T or day as time
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("t", None) or cols.get("day", None)
    incI_col = cols.get("inc_i", None)
    if time_col is None or incI_col is None:
        # Not a summary table we can use
        continue

    # Build one cascade by randomly assigning the inc_I infections each day
    susceptible = set(all_nodes)
    infected_time = {}

    # If header says init_I=1, assign one random seed at time 0.
    init_infections = int(df.iloc[0][incI_col]) if "inc_I" in df.columns else 1
    # Some tables have inc_I=0 at t=0, but header says init_I=1; ensure at least one seed.
    if init_infections <= 0:
        init_infections = 1

    times = df[time_col].to_list()
    incs = df[incI_col].astype(int).to_list()

    for t, k in zip(times, incs):
        k = int(k)
        if k <= 0 or len(susceptible) == 0:
            continue
        choose = rng.choice(list(susceptible), size=min(k, len(susceptible)), replace=False)
        for v in choose:
            infected_time[v] = t
        susceptible.difference_update(choose)

    # if nothing got assigned (edge case), skip this cascade
    if not infected_time:
        continue

    for v, tt in infected_time.items():
        rows.append((cascade_id, str(v), float(tt)))
    cascade_id += 1

if not rows:
    raise SystemExit("Failed to synthesize cascades from counts. Check that the CSVs have inc_I and time/day columns.")

out_df = pd.DataFrame(rows, columns=["cascade_id","node_id","infection_time"])
out_df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(out_df)} records across {cascade_id} cascades.")
