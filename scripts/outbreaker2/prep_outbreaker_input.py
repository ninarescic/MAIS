# scripts/outbreaker2/prep_outbreaker_input.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def to_date_series(infection_times: pd.Series, origin: str) -> pd.Series:
    s = pd.to_datetime(origin) + pd.to_timedelta(infection_times.round().astype(int), unit="D")
    return s.dt.normalize()

def main():
    ap = argparse.ArgumentParser(description="Prepare linelist (id,date) from MAIS cascades")
    ap.add_argument("--cascades", required=True, help="CSV with columns: cascade_id,node_id,infection_time")
    ap.add_argument("--out", required=True, help="Output CSV with columns: id,date")
    ap.add_argument("--origin", default="2020-01-01", help="Origin date (YYYY-MM-DD) for day 0")
    ap.add_argument("--mode", choices=["min", "first"], default="min",
    help="How to aggregate per node over cascades: 'min' infection_time or 'first' by earliest date")
    args = ap.parse_args()


    cascades = Path(args.cascades)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(cascades)
    required = {"cascade_id", "node_id", "infection_time"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Missing columns. Need {required}, got {set(df.columns)}")


    df["node_id"] = df["node_id"].astype(str)
    # aggregate per node: earliest infection time observed across cascades
    agg = df.groupby("node_id")["infection_time"].min().reset_index() if args.mode == "min" else \
        df.sort_values("infection_time").drop_duplicates("node_id")[["node_id", "infection_time"]]


    linelist = pd.DataFrame({
        "id": agg["node_id"].astype(str),
        "date": to_date_series(agg["infection_time"], origin=args.origin)
    })
    linelist.sort_values("date", inplace=True)
    linelist.to_csv(out, index=False)
    print(f"Wrote linelist for {len(linelist)} cases -> {out}")

if __name__ == "__main__":
    main()