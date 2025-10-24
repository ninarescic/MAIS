#!/usr/bin/env python3
"""
Build outbreaker inputs **directly from MAIS** history ZIP/CSV files.

MAIS history format (as in your uploads):
- Text header with lines starting '#'
- Then a CSV header like:
    T,S,I,R,EXT,inc_S,inc_I,inc_R,inc_EXT,day,id
- Rows per day with aggregate counts and new incidences (inc_I),
  and an 'id' string naming the run.

This script EXPANDS daily incidence (default: inc_I) into individual cases:
  case_id = <run_tag>_<zero-padded-seq>
  date    = day (integer)

Outputs:
- dates.csv        columns: id,date          (numeric days from 0)
- w_dens.csv       column:  w                (probability vector, sums to 1)
- contacts.csv     columns: from,to          (optional if inferable)
- sequences.fasta  copied if --fasta provided

Usage (PowerShell from project root):
  python scripts/outbreaker2/prep_outbreaker_input_from_mais.py `
    --hist ./data/output/model/history_*.zip `
    --out_dir ./data/output/model/outbreaker_inputs `
    --inc_col inc_I `
    --w_mean 5 --w_sd 2 `
    --fasta ./data/output/model/sequences.fasta

Notes:
- If you prefer another incidence (e.g., inc_EXT), pass --inc_col inc_EXT.
- If your histories are plain CSVs instead of ZIPs, the same --hist glob works.
"""

import argparse
import io
import sys
import zipfile
import pathlib
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ---------- Robust readers ----------

def _read_text_table_skipping_hash(fb: bytes,
                                   sep: Optional[str] = None) -> pd.DataFrame:
    """
    Read a CSV/TSV text buffer that starts with comment lines '#...'.
    Tries auto-delim first (engine='python'), then fallbacks.
    """
    text = fb.decode("utf-8-sig", errors="replace")
    # Drop comment lines (starting with '#') and empty lines until header
    filtered = "\n".join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))

    # Try auto-detection
    tries = []
    if sep is None:
        tries.append(dict(sep=None, engine="python"))
        tries.append(dict(sep=","))
        tries.append(dict(sep=";"))
        tries.append(dict(sep="\t"))
    else:
        tries.append(dict(sep=sep))

    for kw in tries:
        try:
            df = pd.read_csv(io.StringIO(filtered), on_bad_lines="skip", **kw)
            if isinstance(df, pd.DataFrame) and df.shape[0] >= 1 and df.shape[1] >= 2:
                return df
        except Exception:
            continue
    raise ValueError("Could not parse table after skipping '#' header.")


def read_history_any(path: pathlib.Path) -> pd.DataFrame:
    """
    Read a MAIS history from a .zip (containing a single CSV) or a .csv/.txt file.
    Skips '#' comment header and parses the main table.
    """
    suf = path.suffix.lower()
    if suf == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            # Prefer largest CSV-like member
            members = [m for m in z.infolist() if m.filename.lower().endswith((".csv", ".tsv", ".txt"))]
            if not members:
                raise ValueError(f"No CSV/TSV in zip: {path.name}")
            members.sort(key=lambda m: m.file_size, reverse=True)
            last_exc = None
            for m in members:
                try:
                    with z.open(m, "r") as f:
                        fb = f.read()
                    return _read_text_table_skipping_hash(fb)
                except Exception as e:
                    last_exc = e
                    continue
            raise ValueError(f"Failed to parse any member in {path.name}: {last_exc}")
    elif suf in (".csv", ".tsv", ".txt"):
        fb = path.read_bytes()
        return _read_text_table_skipping_hash(fb)
    else:
        raise ValueError(f"Unsupported history format: {path}")


# ---------- Transform to outbreaker inputs ----------

def expand_incidence_to_cases(df: pd.DataFrame,
                              inc_col: str,
                              run_tag: str,
                              day_col: str = "day",
                              start_seq: int = 1) -> pd.DataFrame:
    """
    Expand a time series DataFrame with columns [<inc_col>, day] into per-case rows:
      id = <run_tag>_<zero-padded-seq>
      date = day (int)
    """
    if inc_col not in df.columns:
        raise KeyError(f"Column '{inc_col}' not found. Available: {list(df.columns)}")
    if day_col not in df.columns:
        # be a bit forgiving on day column casing
        lower = {c.lower(): c for c in df.columns}
        if "day" in lower:
            day_col = lower["day"]
        else:
            raise KeyError(f"Column '{day_col}' not found. Available: {list(df.columns)}")

    # ensure integers
    df = df.copy()
    df[inc_col] = pd.to_numeric(df[inc_col], errors="coerce").fillna(0).astype(int)
    df[day_col] = pd.to_numeric(df[day_col], errors="coerce").fillna(0).astype(int)

    rows: List[Tuple[str, int]] = []
    seq = start_seq
    for _, r in df.iterrows():
        n = int(r[inc_col])
        if n <= 0:
            continue
        d = int(r[day_col])
        for _k in range(n):
            # pad to 4 digits; adjust if you expect many more cases
            cid = f"{run_tag}_{seq:04d}"
            rows.append((cid, d))
            seq += 1

    return pd.DataFrame(rows, columns=["id", "date"])


def discretize_gamma(mean: float, sd: float, max_days: int = 40) -> np.ndarray:
    """Discrete Gamma pmf over days 1..max_days (normalized)."""
    shape = (mean / sd) ** 2
    scale = (sd ** 2) / mean
    x = np.arange(1, max_days + 1, dtype=float)
    # gamma pdf sampled at integers
    from math import gamma
    pdf = (x ** (shape - 1)) * np.exp(-x / scale) / (gamma(shape) * (scale ** shape))
    pmf = pdf / pdf.sum()
    return pmf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True, help="Glob for MAIS histories, e.g. ./data/output/model/history_*.zip")
    ap.add_argument("--out_dir", required=True, help="Output directory for outbreaker inputs")
    ap.add_argument("--inc_col", default="inc_I", help="Incidence column to expand (default: inc_I)")
    ap.add_argument("--case_prefix", default=None,
                    help="Optional fixed prefix for case IDs; default is the run 'id' value from the file")
    ap.add_argument("--w_csv", default=None, help="Optional CSV with column 'w' (probability vector)")
    ap.add_argument("--w_mean", type=float, default=5.0, help="Gamma mean if building w_dens")
    ap.add_argument("--w_sd", type=float, default=2.0, help="Gamma sd if building w_dens")
    ap.add_argument("--w_max_days", type=int, default=40, help="Support length for w_dens")
    ap.add_argument("--fasta", default=None, help="Optional sequences.fasta (IDs must match dates.csv after expansion)")
    ap.add_argument("--contacts_cols", default=None,
                    help="Optional explicit contacts columns as from:to (rare in these histories)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect all history paths
    hist_paths = sorted(pathlib.Path().glob(args.hist))
    if not hist_paths:
        print(f"No files match: {args.hist}", file=sys.stderr)
        sys.exit(2)

    # accumulate per-case rows from all histories
    all_dates: List[pd.DataFrame] = []
    for hp in hist_paths:
        try:
            df = read_history_any(hp)
        except Exception as e:
            print(f"[WARN] skipping {hp.name}: {e}", file=sys.stderr)
            continue

        # derive a run tag
        run_tag = args.case_prefix
        if run_tag is None:
            # If the table has an 'id' column with a single value repeated, use it; else use filename stem
            run_tag = None
            if "id" in df.columns:
                uniq = pd.unique(df["id"].astype(str).fillna(""))
                uniq = [u for u in uniq if u.strip()]
                if len(uniq) == 1:
                    run_tag = uniq[0]
            if not run_tag:
                run_tag = hp.stem  # filename without .zip/.csv

        dates_df = expand_incidence_to_cases(df, inc_col=args.inc_col, run_tag=run_tag, day_col="day")
        if dates_df.empty:
            print(f"[WARN] {hp.name}: no cases produced from '{args.inc_col}'.", file=sys.stderr)
        all_dates.append(dates_df)

    if not all_dates:
        print("No readable histories produced cases. Aborting.", file=sys.stderr)
        sys.exit(2)

    dates = pd.concat(all_dates, ignore_index=True)

    # normalize date origin to 0 across all runs
    dates["date"] = pd.to_numeric(dates["date"], errors="coerce").astype(int)
    dates["date"] = dates["date"] - dates["date"].min()

    # write dates.csv
    dates_out = out_dir / "dates.csv"
    dates.sort_values(["date", "id"]).to_csv(dates_out, index=False)
    print(f"WROTE: {dates_out}")

    # write w_dens.csv
    if args.w_csv:
        w_df = pd.read_csv(args.w_csv)
        if "w" not in w_df.columns:
            raise ValueError("--w_csv must contain a column named 'w'")
        w = w_df["w"].astype(float).to_numpy()
        w = w / w.sum()
    else:
        w = discretize_gamma(args.w_mean, args.w_sd, args.w_max_days)
    w_out = out_dir / "w_dens.csv"
    pd.DataFrame({"w": w}).to_csv(w_out, index=False)
    print(f"WROTE: {w_out}")

    # contacts.csv (only if explicitly present or future extension)
    if args.contacts_cols:
        from_col, to_col = (c.strip() for c in args.contacts_cols.split(":", 1))
        # Try to read again (just to search for these cols); will write only if both exist
        contacts_rows: List[pd.DataFrame] = []
        for hp in hist_paths:
            try:
                df = read_history_any(hp)
            except Exception:
                continue
            if from_col in df.columns and to_col in df.columns:
                tmp = df[[from_col, to_col]].dropna().astype(str)
                tmp.columns = ["from", "to"]
                # Keep only IDs that exist in dates
                ids = set(dates["id"])
                tmp = tmp[tmp["from"].isin(ids) & tmp["to"].isin(ids)]
                if not tmp.empty:
                    contacts_rows.append(tmp)
        if contacts_rows:
            contacts = pd.concat(contacts_rows, ignore_index=True).drop_duplicates()
            c_out = out_dir / "contacts.csv"
            contacts.to_csv(c_out, index=False)
            print(f"WROTE: {c_out}")

    # sequences.fasta passthrough
    if args.fasta:
        src = pathlib.Path(args.fasta)
        if src.exists():
            dst = out_dir / "sequences.fasta"
            dst.write_bytes(src.read_bytes())
            print(f"WROTE: {dst}")
        else:
            print(f"[WARN] FASTA not found: {src}", file=sys.stderr)


if __name__ == "__main__":
    main()
