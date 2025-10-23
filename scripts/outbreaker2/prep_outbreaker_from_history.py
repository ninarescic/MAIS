# scripts/outbreaker2/prep_outbreaker_from_history.py
"""
Create a linelist (id,date) for ONE outbreak directly from MAIS outputs,
NOT from NetRate cascades.

It will look for per-node state/event data either:
  • inside a single history ZIP (recommended: one run/seed), or
  • in a plain CSV with per-node states.

Heuristics supported:
  A) "node_states" files with columns like: node_id/id, day/t, state
     -> infection_time = min day where state == 'I' (configurable)
  B) per-node event file with 'infection_time'
     -> use that directly

Usage examples (from project root):

1) Single ZIP run (one outbreak):
   python -m scripts.outbreaker2.prep_outbreaker_from_history \
     --input data/output/model/history_netrate_agent_test_MODEL_beta=0.5_MODEL_I_duration=7_0.zip \
     --out   data/output/model/outbreaker2/linelist.csv \
     --origin 2020-01-01 --infected_label I --id_offset 0

2) Plain CSV per-node states:
   python -m scripts.outbreaker2.prep_outbreaker_from_history \
     --input data/output/model/node_states_netrate_agent_test_seed1.csv \
     --out   data/output/model/outbreaker2/linelist.csv

IMPORTANT:
 • Use ONE outbreak (one run/seed) per linelist.
 • If Verona node ids start at 1 and your logs start at 0, set --id_offset 1.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import zipfile


def to_dates(infection_times: pd.Series, origin: str) -> pd.Series:
    s = pd.to_datetime(origin) + pd.to_timedelta(infection_times.round().astype(int), unit="D")
    return s.dt.normalize()


def _infer_columns(df: pd.DataFrame):
    """Try to detect column names for node id, time, state, infection_time."""
    cols = {c.lower(): c for c in df.columns}

    # direct infection_time path
    inf_col = None
    for key in ("infection_time", "infected_at", "t_inf", "t_infection"):
        if key in cols:
            inf_col = cols[key]
            break

    # node id
    node_col = None
    for key in ("node_id", "id", "node", "agent_id"):
        if key in cols:
            node_col = cols[key]
            break

    # time/day
    time_col = None
    for key in ("day", "t", "time"):
        if key in cols:
            time_col = cols[key]
            break

    # state label
    state_col = None
    for key in ("state", "status"):
        if key in cols:
            state_col = cols[key]
            break

    return node_col, time_col, state_col, inf_col


def linelist_from_df(df: pd.DataFrame, infected_label: str, origin: str, id_offset: int) -> pd.DataFrame:
    node_col, time_col, state_col, inf_col = _infer_columns(df)

    if inf_col:
        # Already has infection times
        tmp = df[[node_col, inf_col]].dropna().copy()
        tmp[node_col] = tmp[node_col].astype(int) + id_offset
        tmp["id"] = tmp[node_col].astype(str)
        tmp["date"] = to_dates(tmp[inf_col], origin)
        out = tmp[["id", "date"]].drop_duplicates("id")
        return out.sort_values("date")

    # Otherwise, derive infection time as first time state == infected_label
    if not (node_col and time_col and state_col):
        raise ValueError(
            f"Could not infer required columns. Found: node={node_col}, time={time_col}, state={state_col}, infection_time={inf_col}"
        )

    df = df[[node_col, time_col, state_col]].copy()
    df[state_col] = df[state_col].astype(str)

    infected = df[df[state_col] == infected_label]
    if infected.empty:
        raise ValueError(f"No rows with state=='{infected_label}' found.")

    # earliest infection time per node
    agg = infected.groupby(node_col, as_index=False)[time_col].min()
    agg[node_col] = agg[node_col].astype(int) + id_offset

    out = pd.DataFrame(
        {
            "id": agg[node_col].astype(str),
            "date": to_dates(agg[time_col], origin),
        }
    ).sort_values("date")
    return out


def read_first_node_table_from_zip(zpath: Path) -> pd.DataFrame:
    """
    Open a MAIS history ZIP and return the first CSV that looks like per-node data.
    Robust to commented headers ('#') and non-comma delimiters (space/semicolon).
    """
    import io, pandas as pd, zipfile

    def try_read_csv(filelike):
        # 1) fast path: comma CSV, ignore comments
        try:
            return pd.read_csv(filelike, comment="#")
        except Exception:
            pass
        # 2) sniff delimiter with Python engine (handles spaces, tabs, semicolons)
        filelike.seek(0)  # rewind
        try:
            return pd.read_csv(filelike, sep=None, engine="python", comment="#")
        except Exception:
            return None

    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]

        # Prefer explicit per-node files first
        priority = [
            n for n in names
            if any(k in n.lower() for k in ("node_states", "per_node", "nodes_states", "node-state", "node_state"))
        ]
        candidates = priority + [n for n in names if n not in priority]

        for name in candidates:
            with z.open(name) as f:
                # read into a BytesIO so we can seek between attempts
                buf = io.BytesIO(f.read())
            df = try_read_csv(buf)
            if df is None or df.empty:
                continue

            # Check if this CSV has columns that allow deriving infection times
            node_col, time_col, state_col, inf_col = _infer_columns(df)
            if (inf_col is not None) or (node_col and (state_col or time_col)):
                # good candidate
                # print(f"[ZIP] Using '{name}' from {zpath.name}")
                return df

    raise ValueError(
        "No suitable per-node CSV found in ZIP. "
        "Make sure your simulation ran with 'save_node_states = Yes', "
        "or point --input to a per-node CSV directly."
    )



def main():
    ap = argparse.ArgumentParser(description="Build linelist (id,date) from a SINGLE MAIS outbreak (ZIP or CSV)")
    ap.add_argument("--input", required=True, help="Path to a history ZIP from one run/seed OR a per-node CSV.")
    ap.add_argument("--out", required=True, help="Output CSV path (id,date).")
    ap.add_argument("--origin", default="2020-01-01", help="Origin date (YYYY-MM-DD) for day 0.")
    ap.add_argument("--infected_label", default="I", help="State label that marks infection (default: I).")
    ap.add_argument("--id_offset", type=int, default=0, help="Add this offset to node ids (e.g., 1 to convert 0->1-based).")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    if inp.suffix.lower() == ".zip":
        df = read_first_node_table_from_zip(inp)
    else:
        df = pd.read_csv(inp)

    linelist = linelist_from_df(df, infected_label=args.infected_label, origin=args.origin, id_offset=args.id_offset)
    linelist.to_csv(out, index=False)
    print(f"Wrote linelist with {len(linelist)} cases -> {out}")


if __name__ == "__main__":
    main()
