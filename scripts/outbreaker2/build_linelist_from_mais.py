# scripts/outbreaker2/build_linelist_from_mais.py
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile
import io
import pandas as pd

CANDIDATE_NODE_COLS = ["node_id", "id", "node"]
STATE_COL = "state"
DAY_COL = "day"
INFECTION_TIME_COL = "infection_time"  # if present, we trust it

def find_first_infection(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CANDIDATE_NODE_COLS if c in df.columns]
    if not cols:
        raise ValueError("No node id column found")
    node_col = cols[0]

    # Case A: direct infection_time available
    if INFECTION_TIME_COL in df.columns:
        out = (
            df[[node_col, INFECTION_TIME_COL]]
            .dropna(subset=[INFECTION_TIME_COL])
            .groupby(node_col, as_index=False)[INFECTION_TIME_COL]
            .min()
            .rename(columns={node_col: "id", INFECTION_TIME_COL: "onset_day"})
        )
        return out

    # Case B: state/day logs; find first time state == 'I'
    if STATE_COL in df.columns and DAY_COL in df.columns:
        # normalize state to string upper
        s = df[STATE_COL].astype(str).str.upper()
        mask_I = s.eq("I")
        if not mask_I.any():
            return pd.DataFrame(columns=["id", "onset_day"])
        sub = df.loc[mask_I, :]
        out = (
            sub[[cols[0], DAY_COL]]
            .groupby(cols[0], as_index=False)[DAY_COL]
            .min()
            .rename(columns={cols[0]: "id", DAY_COL: "onset_day"})
        )
        return out

    raise ValueError("No usable columns (need either infection_time, or state+day).")

def read_zip_first_infections(zpath: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        # Try all CSVs; return the first that yields infections
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        # Heuristic: prefer filenames that suggest per-node states
        pref = sorted(csv_names, key=lambda n: (("node_states" not in n.lower()), n))
        for name in pref:
            with z.open(name) as f:
                try:
                    df = pd.read_csv(io.BytesIO(f.read()))
                except Exception:
                    continue
                try:
                    out = find_first_infection(df)
                    if not out.empty:
                        return out
                except Exception:
                    continue
    # If nothing worked, return empty
    return pd.DataFrame(columns=["id", "onset_day"])

def to_dates(onset_days: pd.Series, origin: str) -> pd.Series:
    return (pd.to_datetime(origin) + pd.to_timedelta(onset_days.round().astype(int), unit="D")).dt.normalize()

def build_one(zpath: Path, out_csv: Path, origin: str, id_offset: int):
    inf = read_zip_first_infections(zpath)
    if inf.empty:
        print(f"[WARN] No infections extracted from {zpath.name}")
        return False
    inf["id"] = (inf["id"].astype(str).str.strip())
    # if numeric, apply offset (e.g., +1 to convert 0-based -> 1-based)
    try:
        inf["id"] = (inf["id"].astype(int) + id_offset).astype(str)
    except Exception:
        # non-numeric ids → just keep as strings
        pass
    linelist = pd.DataFrame({"id": inf["id"], "date": to_dates(inf["onset_day"], origin)})
    linelist = linelist.dropna().drop_duplicates().sort_values("date")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    linelist.to_csv(out_csv, index=False)
    print(f"[OK] Wrote linelist ({len(linelist)} cases) -> {out_csv}")
    return True

def main():
    ap = argparse.ArgumentParser(description="Build Outbreaker linelists directly from MAIS ZIP outputs")
    ap.add_argument("--glob", default="data/output/model/history_*.zip",
                    help="Glob for MAIS output zips (one linelist per zip).")
    ap.add_argument("--out_dir", default="data/output/model/outbreaker2",
                    help="Where to write linelists (one CSV per ZIP).")
    ap.add_argument("--merge_out", default=None,
                    help="If set, also write a merged linelist CSV with all unique cases.")
    ap.add_argument("--origin", default="2020-01-01", help="Date origin for day 0 (YYYY-MM-DD).")
    ap.add_argument("--id_offset", type=int, default=0, help="Add to node IDs (e.g., 1 to convert 0-based -> 1-based).")
    args = ap.parse_args()

    root = Path(".").resolve()
    zips = sorted(Path().glob(args.glob))
    if not zips:
        raise SystemExit(f"No ZIPs matched: {args.glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = []
    for z in zips:
        out_csv = out_dir / f"linelist__{z.stem}.csv"
        ok = build_one(z, out_csv, origin=args.origin, id_offset=args.id_offset)
        if ok:
            df = pd.read_csv(out_csv)
            df["source_zip"] = z.name
            merged.append(df)

    if args.merge_out:
        if merged:
            mdf = pd.concat(merged, ignore_index=True).drop_duplicates()
            mdf.to_csv(Path(args.merge_out), index=False)
            print(f"[OK] Wrote merged linelist ({len(mdf)} rows) -> {args.merge_out}")
        else:
            print("[WARN] Nothing to merge; no linelists were produced.")

if __name__ == "__main__":
    main()
