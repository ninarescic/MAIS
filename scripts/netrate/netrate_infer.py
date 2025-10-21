#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NetRate (exponential kernel) in Python using cvxpy.
Input:  CSV with columns: cascade_id,node_id,infection_time
Output: CSV with columns: source,target,beta
"""

import argparse
from collections import defaultdict
from math import inf
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp


def load_cascades(path_csv: Path):
    """
    Returns:
      cascades: dict[cid] -> dict[node] = infection_time (float)
      nodes: sorted list of all nodes seen
      T_end: dict[cid] -> observation window (max observed infection time in that cascade)
    """
    df = pd.read_csv(path_csv)
    # normalize types
    df["cascade_id"] = df["cascade_id"].astype(str)
    df["node_id"] = df["node_id"].astype(str)
    df["infection_time"] = df["infection_time"].astype(float)

    cascades = defaultdict(dict)
    for cid, nid, t in df[["cascade_id", "node_id", "infection_time"]].itertuples(index=False, name=None):
        cascades[cid][nid] = float(t)

    nodes = sorted(set(df["node_id"].unique().tolist()))
    # observation window: last observed infection time in each cascade
    T_end = {cid: max(times.values()) for cid, times in cascades.items()}
    return cascades, nodes, T_end


def infer_targets(cascades, nodes, T_end, l1=1e-2, thr=1e-8, verbose=True):
    """
    Solve NetRate per target node (exponential transmission).
    Returns a list of (src, dst, beta) for beta > thr.
    """
    results = []
    node_index = {n: i for i, n in enumerate(nodes)}

    # Precompute for speed: per cascade, list of infected nodes and their times
    casc_nodes = {}
    for cid, times in cascades.items():
        items = sorted(times.items(), key=lambda kv: kv[1])
        casc_nodes[cid] = (np.array([n for n, _ in items], dtype=object),
                           np.array([t for _, t in items], dtype=float))

    for idx_i, i in enumerate(nodes):
        if verbose:
            print(f"[NetRate] Target {idx_i+1}/{len(nodes)}: {i}")

        # Candidate parents j: any node that was infected in any cascade before either t_i^c (if i got infected)
        # or before T_c (if i never got infected in that cascade). Union over cascades.
        parents = set()
        for cid, (n_arr, t_arr) in casc_nodes.items():
            t_i = cascades[cid].get(i, inf)
            limit = t_i if np.isfinite(t_i) else T_end[cid]
            # add nodes infected before 'limit'
            for n_j, t_j in zip(n_arr, t_arr):
                if t_j < limit and n_j != i:
                    parents.add(n_j)

        parents = sorted(parents)
        if not parents:
            continue  # no potential parents -> no incoming edges inferred

        K = len(parents)
        b = cp.Variable(K, nonneg=True)  # beta_{j->i}, j in parents

        obj_terms = []

        # Build objective contributions cascade by cascade
        for cid, (n_arr, t_arr) in casc_nodes.items():
            t_i = cascades[cid].get(i, inf)

            # mask over parents for those who were infected before the relevant time
            # create vectors aligned with 'parents'
            t_j_vec = np.full(K, np.nan)
            for k, pj in enumerate(parents):
                # infection time of parent pj in cascade cid (nan if never infected)
                t_j = cascades[cid].get(pj, np.nan)
                t_j_vec[k] = t_j

            if np.isfinite(t_i):
                # i infected in this cascade
                mask = ~np.isnan(t_j_vec) & (t_j_vec < t_i)
                if not np.any(mask):
                    continue  # nothing contributed by this cascade
                delta = (t_i - t_j_vec[mask])  # positive durations
                # Linear term: sum_j beta_j * (t_i - t_j)
                lin = cp.sum(cp.multiply(b[mask], delta))
                # -log( sum_j beta_j )
                # Guard for numerical stability with tiny epsilon inside log
                log_term = -cp.log(cp.sum(b[mask]) + 1e-12)
                obj_terms.append(lin + log_term)
            else:
                # i NOT infected: only survival until observation window T_c
                T = T_end[cid]
                mask = ~np.isnan(t_j_vec) & (t_j_vec < T)
                if not np.any(mask):
                    continue
                delta = (T - t_j_vec[mask])
                lin = cp.sum(cp.multiply(b[mask], delta))
                obj_terms.append(lin)

        if not obj_terms:
            # Nothing informative for this target
            continue

        objective = cp.Minimize(cp.sum(obj_terms) + l1 * cp.norm1(b))

        prob = cp.Problem(objective)
        try:
            prob.solve(solver=cp.SCS, verbose=False)  # SCS is robust; you can try ECOS/OSQP too
        except Exception as e:
            if verbose:
                print(f"  -> Solver failed for target {i}: {e}")
            continue

        if b.value is None:
            continue

        betas = np.asarray(b.value).ravel()
        for pj, beta in zip(parents, betas):
            if beta is not None and beta > thr:
                results.append((pj, i, float(beta)))

    return results


def main():
    ap = argparse.ArgumentParser(description="NetRate (exponential) in Python with cvxpy.")
    ap.add_argument("--cascades", required=True, help="Path to netrate_cascades.csv")
    ap.add_argument("--l1", type=float, default=1e-2, help="L1 sparsity weight (lambda)")
    ap.add_argument("--thr", type=float, default=1e-8, help="Threshold to keep edges")
    ap.add_argument("--out", default="netrate_result.csv", help="Output CSV for inferred edges")
    args = ap.parse_args()

    cascades, nodes, T_end = load_cascades(Path(args.cascades))
    print(f"Loaded {len(cascades)} cascades, {len(nodes)} nodes.")

    edges = infer_targets(cascades, nodes, T_end, l1=args.l1, thr=args.thr, verbose=True)
    if not edges:
        print("No edges inferred (all betas ~ 0). Try reducing --l1 or providing more cascades.")
        return

    out_df = pd.DataFrame(edges, columns=["source", "target", "beta"])
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} edges to {out_path.resolve()}")


if __name__ == "__main__":
    main()
