# scripts/outbreaker2/py_outbreaker_infer.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gamma as sgamma



def discretized_gamma_pmf(max_days: int, shape: float, scale: float) -> np.ndarray:
    x = np.arange(0, max_days + 1, dtype=float)
    cdf = sgamma.cdf(x, a=shape, scale=scale)
    pmf = np.diff(cdf) # length max_days; index k-1 corresponds to Δt=k
    pmf = np.maximum(pmf, np.finfo(float).eps)

    pmf /= pmf.sum()
    return pmf

def map_links(linelist: pd.DataFrame, max_days: int, shape: float, scale: float,
    import_prior: float = 0.1) -> pd.DataFrame:
    """
    For each child, consider parents with earlier onset. Score by PMF(Δt).
    Add an 'importation' option with prior probability; if all candidates are weak,
    mark as imported (no edge).
    Returns DataFrame: source,target,support (support in [0,1], per-child normalized over candidates).
    """
    linelist = linelist.copy()
    linelist["id"] = linelist["id"].astype(str)
    linelist["date"] = pd.to_datetime(linelist["date"]).dt.normalize()


    ids = linelist["id"].tolist()
    t = linelist.set_index("id")["date"]

    w = discretized_gamma_pmf(max_days=max_days, shape=shape, scale=scale) # w[k-1] = P(Δt=k)

    edges = []
    for child in ids:
        # candidate parents: onset strictly earlier than child
        cand = [p for p in ids if p != child and t[p] < t[child]]
        if not cand:
            continue # imported
        dts = np.array([(t[child] - t[p]).days for p in cand], dtype=int)
        mask = (dts >= 1) & (dts <= max_days)
        cand = [c for c, m in zip(cand, mask) if m]
        dts = dts[mask]
        if len(cand) == 0:
            continue # imported


        # likelihoods for each candidate parent
        L = np.array([w[dt - 1] for dt in dts], dtype=float)
        # include an import option with prior weight
        L_import = import_prior
        total = L.sum() + L_import
        if total <= 0:
            continue
        # normalized supports
        supports = L / total
        # pick MAP parent
        j = int(np.argmax(L))
        edges.append((cand[j], child, float(supports[j])))

    return pd.DataFrame(edges, columns=["source", "target", "support"]).sort_values("support", ascending=False)

def main():
    ap = argparse.ArgumentParser(description="Python MAP-style outbreaker (no MCMC)")
    ap.add_argument("--linelist", required=True, help="CSV with columns: id,date")
    ap.add_argument("--out", required=True, help="Output CSV path (source,target,support)")
    ap.add_argument("--max_days", type=int, default=30, help="Max plausible generation interval (days)")
    ap.add_argument("--gt_shape", type=float, default=2.5, help="Gamma shape for generation time")
    ap.add_argument("--gt_scale", type=float, default=1.9, help="Gamma scale for generation time")
    ap.add_argument("--import_prior", type=float, default=0.1, help="Prior mass for importation (no parent)")
    args = ap.parse_args()


    linelist = pd.read_csv(args.linelist)
    need = {"id", "date"}
    if not need.issubset(linelist.columns):
        raise SystemExit(f"Missing columns. Need {need}, got {set(linelist.columns)}")


    out_df = map_links(linelist, args.max_days, args.gt_shape, args.gt_scale, args.import_prior)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} links -> {out_path}")

if __name__ == "__main__":
    main()