# scripts/extract_netrate_cascades.py
from pathlib import Path
import zipfile
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "output" / "model"
OUT_CSV = OUT_DIR / "netrate_cascades.csv"

# Only pick the sweep you showed
# Only pick the agent-level zips you just created
zips = sorted(OUT_DIR.glob("history_netrate_agent_test_MODEL_*.zip"))
if not zips:
    zips = sorted(OUT_DIR.glob("history_*.zip"))


def safe_read(file_like):
    # auto-detect delimiters; skip malformed rows
    return pd.read_csv(file_like, sep=None, engine="python", on_bad_lines="skip")

def lower_cols(df):
    return df.rename(columns={c: c.strip().lower() for c in df.columns})

def looks_per_node(df):
    cols = set(df.columns)
    has_node = any(c in cols for c in ["node_id","node","id","agent"])
    has_time = any(c in cols for c in ["time","t","step"])
    has_state = any(c in cols for c in ["state","status","compartment","to_state","to"])
    return has_node and has_time and has_state

def normalize(df):
    df = lower_cols(df)
    # map node column
    for c in ["node_id","node","id","agent"]:
        if c in df.columns:
            df = df.rename(columns={c:"node_id"})
            break
    # map time column
    for c in ["time","t","step"]:
        if c in df.columns:
            df = df.rename(columns={c:"time"})
            break
    # prefer explicit transition target if present
    if "to_state" in df.columns:
        df = df.rename(columns={"to_state":"state"})
    elif "to" in df.columns:
        df = df.rename(columns={"to":"state"})
    return df

def infection_times_from(df):
    df = normalize(df)
    if not {"node_id","time"}.issubset(df.columns):
        raise ValueError("missing node_id/time")
    # choose a state column
    state_col = next((c for c in ["state","status","compartment"] if c in df.columns), None)
    if state_col is None:
        raise ValueError("missing state")

    # mark infected labels
    infected_labels = {"i","infected","inf","ill","symptomatic","active"}
    s = df[state_col].astype(str).str.strip().str.lower()
    df_inf = df[s.isin(infected_labels)].copy()
    if df_inf.empty:
        raise ValueError("no infected rows")
    first = (df_inf.groupby("node_id", as_index=False)["time"]
                  .min().rename(columns={"time":"infection_time"}))
    return first

def iter_csvs_in_zip(zp: Path):
    with zipfile.ZipFile(zp) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                try:
                    df = safe_read(f)
                except Exception:
                    continue
            df = lower_cols(df)
            if looks_per_node(df):
                yield name, df

rows = []
for z in zips:
    # Use index in list as cascade id (or parse from filename if you prefer)
    m = re.search(r"_(\d+)\.zip$", z.name)
    cascade_id = int(m.group(1)) if m else len(rows)

    # Some ZIPs have multiple CSVs; combine all that look per-node
    parts = []
    for name, df in iter_csvs_in_zip(z):
        parts.append(df)
    if not parts:
        continue
    df_all = pd.concat(parts, ignore_index=True)

    try:
        inf = infection_times_from(df_all)
    except Exception:
        # Try again using only rows that explicitly have transitions to I
        # (handles logs with from_state/to_state)
        if {"to_state","to"}.intersection(df_all.columns):
            df2 = df_all.copy()
            col = "to_state" if "to_state" in df2.columns else "to"
            s = df2[col].astype(str).str.lower()
            df2 = df2[s.isin({"i","infected","inf"})]
            if {"node_id"}.issubset(df2.columns) and any(c in df2.columns for c in ["time","t","step"]):
                for c in ["time","t","step"]:
                    if c in df2.columns: df2 = df2.rename(columns={c:"time"}); break
                inf = (df2.groupby("node_id", as_index=False)["time"]
                          .min().rename(columns={"time":"infection_time"}))
            else:
                continue
        else:
            continue

    inf.insert(0, "cascade_id", cascade_id)
    rows.append(inf)

if not rows:
    raise SystemExit("Still no usable cascades. (We matched only per-node CSVs. If your ZIPs only contain summaries, we need a different output mode.)")

all_inf = pd.concat(rows, ignore_index=True)
all_inf["node_id"] = all_inf["node_id"].astype(str)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
all_inf[["cascade_id","node_id","infection_time"]].to_csv(OUT_CSV, index=False)
print(f"Saved NetRate cascades to: {OUT_CSV}")
print(all_inf.head())
