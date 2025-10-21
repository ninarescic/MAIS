#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Discrete SIR runner with NetRate per-agent output.

- Reads Verona graph (or what the INI points to)
- Supports multiple betas and multiple I_duration values in the INI:
    [MODEL]
    beta = 0.2;0.5
    I_duration = 2;7
- Produces:
    1) history ZIPs like: data/output/model/history_<name>_MODEL_beta=..._MODEL_I_duration=... .zip
    2) NetRate per-agent CSVs like:
       data/output/model/netrate/netrate_cascade_MODEL_beta=..._MODEL_I_duration=..._seed=... .csv

Usage:
  python run_experiment_netrate.py -r ../config/verona_sir.ini netrate_agent_test
"""

from __future__ import annotations

import argparse
import configparser
import csv
import io
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


# ------------------------- config helpers -------------------------

def _yes(s: str | bool | None) -> bool:
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_num_list(s: str, typ=float) -> List:
    """
    Parse 'a;b;c' or 'a,b,c' or single 'a' into a list of numbers (float/int).
    """
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    s = s.replace(",", ";")
    vals = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(typ(tok))
    return vals


@dataclass
class Settings:
    nodes_csv: Path
    edges_csv: Path
    betas: List[float]
    I_durations: List[int]
    init_I: int
    duration: int
    out_dir: Path
    output_id_base: str
    seed: int
    print_interval: int
    verbose: bool


def load_settings(ini_path: Path) -> Settings:
    cp = configparser.ConfigParser()
    cp.read(ini_path)

    nodes_csv = Path(cp.get("GRAPH", "nodes", fallback="../data/m-input/verona/raj-nodes.csv"))
    edges_csv = Path(cp.get("GRAPH", "edges", fallback="../data/m-input/verona/raj-full-edges.csv"))

    # MODEL
    beta_raw = cp.get("MODEL", "beta", fallback="0.2")
    I_raw = cp.get("MODEL", "I_duration", fallback="7")
    betas = _parse_num_list(beta_raw, float) or [0.2]
    I_durations = _parse_num_list(I_raw, int) or [7]
    init_I = cp.getint("MODEL", "init_I", fallback=1)

    # TASK
    duration = cp.getint("TASK", "duration_in_days", fallback=30)
    out_dir = Path(cp.get("TASK", "output_dir", fallback="../data/output/model"))
    print_interval = cp.getint("TASK", "print_interval", fallback=1)
    verbose = _yes(cp.get("TASK", "verbose", fallback="Yes"))
    seed = cp.getint("TASK", "random_seed", fallback=16)

    # OUTPUT
    output_id_base = cp.get("OUTPUT_ID", "id", fallback="")

    return Settings(
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        betas=betas,
        I_durations=I_durations,
        init_I=init_I,
        duration=duration,
        out_dir=out_dir,
        output_id_base=output_id_base,
        seed=seed,
        print_interval=print_interval,
        verbose=verbose,
    )


# ------------------------- IO utils -------------------------

def _read_csv_to_dicts(p: Path) -> List[Dict[str, str]]:
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return [r for r in rdr]


def load_graph(nodes_csv: Path, edges_csv: Path) -> nx.Graph:
    import pandas as pd
    import re

    def norm_raw(x: str) -> str:
        return str(x).strip().lower()

    def norm_nozeros(x: str) -> str:
        s = norm_raw(x)
        return s.lstrip("0")

    def norm_dropn(x: str) -> str:
        s = norm_raw(x)
        return s[1:] if s.startswith("n") else s

    def norm_digits(x: str) -> str:
        s = "".join(ch for ch in norm_raw(x) if ch.isdigit())
        return s

    NORMALIZERS = [norm_raw, norm_nozeros, norm_dropn, norm_digits]

    nd = pd.read_csv(nodes_csv, dtype=str)
    nd.columns = [c.strip() for c in nd.columns]
    node_col = next((c for c in nd.columns if c.lower() in {"id","node","node_id","name","vertex","index","uid"}), nd.columns[0])

    ed = pd.read_csv(edges_csv, dtype=str)
    ed.columns = [c.strip() for c in ed.columns]

    endpoint_pairs = [
        ("vertex1", "vertex2"),  # ✅ Verona dataset
        ("u", "v"), ("src", "dst"), ("source", "target"), ("from", "to"),
        ("a", "b"), ("node1", "node2"), ("n1", "n2"),
    ]
    if len(ed.columns) >= 2:
        endpoint_pairs.append((ed.columns[0], ed.columns[1]))
    endpoint_pairs = [p for p in endpoint_pairs if all(c in ed.columns for c in p)]

    best = None; best_edges = -1; best_info = None
    for norm_nodes in NORMALIZERS:
        node_ids = nd[node_col].astype(str).map(norm_nodes)
        node_set = set(node_ids)

        for ucol, vcol in endpoint_pairs:
            u_raw = ed[ucol].astype(str); v_raw = ed[vcol].astype(str)
            for norm_edges in NORMALIZERS:
                u = u_raw.map(norm_edges); v = v_raw.map(norm_edges)
                mask = u.isin(node_set) & v.isin(node_set) & (u != v)
                n_edges = int(mask.sum())
                if n_edges > best_edges:
                    best_edges = n_edges
                    best = (node_set, list(zip(u[mask], v[mask])))
                    best_info = dict(node_col=node_col, ucol=ucol, vcol=vcol,
                                     node_norm=norm_nodes.__name__, edge_norm=norm_edges.__name__)

    G = nx.Graph()
    if best is None:
        raise RuntimeError("Could not align nodes/edges; please inspect the CSV headers/values.")
    node_set, edge_list = best
    G.add_nodes_from(node_set)
    G.add_edges_from(edge_list)

    print(f"[GRAPH] chosen node_col='{best_info['node_col']}', edge_cols=('{best_info['ucol']}','{best_info['vcol']}'), "
          f"node_norm={best_info['node_norm']}, edge_norm={best_info['edge_norm']}")
    print(f"[GRAPH] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G



# ------------------------- SIR simulator -------------------------

@dataclass
class SIRState:
    S: set
    I: set
    R: set
    i_clock: Dict[str, int]
    infection_time: Dict[str, float]  # first time a node enters I


def simulate_sir(
    G: nx.Graph,
    beta: float,
    I_duration: int,
    init_I: int,
    duration: int,
    seed: int,
    verbose: bool,
    print_interval: int,
) -> Tuple[SIRState, List[Dict[str, int]]]:

    rng = random.Random(seed)
    nodes = list(G.nodes())
    if init_I > len(nodes):
        init_I = len(nodes)

    # Initial infection(s) at t=0
    initial = rng.sample(nodes, init_I) if init_I > 0 else []

    S = set(nodes)
    I = set()
    R = set()
    i_clock: Dict[str, int] = {}
    infection_time: Dict[str, float] = {}

    for v in initial:
        if v in S:
            S.remove(v)
        I.add(v)
        i_clock[v] = I_duration
        infection_time[v] = 0.0

    history_rows: List[Dict[str, int]] = []

    for day in range(0, duration + 1):
        # Log counts BEFORE the day's transitions (consistent with many simulators)
        T = len(S) + len(I) + len(R)
        row = {
            "T": T,
            "S": len(S),
            "I": len(I),
            "R": len(R),
            "EXT": 0,
            "inc_S": 0,
            "inc_I": 0,
            "inc_R": 0,
            "inc_EXT": 0,
            "day": day,
        }
        history_rows.append(row)

        if verbose and (day % print_interval == 0):
            print(f"day={day:02d}  S={len(S)} I={len(I)} R={len(R)}")

        if day == duration:
            break

        # Infection phase
        newly_infected = set()
        for u in list(I):
            for v in G.neighbors(u):
                if v in S and rng.random() < beta:
                    newly_infected.add(v)

        # Recovery phase (countdown clocks)
        to_recover = set()
        for u in list(I):
            i_clock[u] -= 1
            if i_clock[u] <= 0:
                to_recover.add(u)

        # Apply infections
        for v in newly_infected:
            if v in S:
                S.remove(v)
                I.add(v)
                i_clock[v] = I_duration
                if v not in infection_time:
                    infection_time[v] = float(day + 1)  # becomes I after this step

        # Apply recoveries
        for u in to_recover:
            if u in I:
                I.remove(u)
                R.add(u)
                del i_clock[u]

        # Update increments for the row we logged at 'day'
        row["inc_I"] = len(newly_infected)
        row["inc_R"] = len(to_recover)

    state = SIRState(S=S, I=I, R=R, i_clock=i_clock, infection_time=infection_time)
    return state, history_rows


# ------------------------- writers -------------------------

def write_history_zip(out_dir: Path, tag: str, header_meta: Dict[str, str], history_rows: List[Dict[str, int]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"{tag}_0.csv"
    zip_name = f"{tag}.zip"
    zip_path = out_dir / zip_name

    sio = io.StringIO()
    # Commented header (matches your existing style)
    sio.write("#[GRAPH]\n")
    for k, v in header_meta.items():
        sio.write(f"#{k} = {v}\n")
    sio.write("\n")

    # Table
    fieldnames = ["T", "S", "I", "R", "EXT", "inc_S", "inc_I", "inc_R", "inc_EXT", "day", "id"]
    w = csv.DictWriter(sio, fieldnames=fieldnames)
    w.writeheader()
    for row in history_rows:
        r = dict(row)
        r["id"] = tag
        w.writerow(r)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(csv_name, sio.getvalue().encode("utf-8"))
    return zip_path


def write_netrate_csv(out_dir: Path, output_id: str, seed: int, infection_time: Dict[str, float]) -> Path:
    p = out_dir / "netrate"
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"netrate_cascade{output_id}_seed={seed}.csv"
    with out.open("w", encoding="utf-8") as f:
        f.write("cascade_id,node_id,infection_time\n")
        cid = f"{output_id}_seed={seed}"
        for v, tt in infection_time.items():
            f.write(f"{cid},{v},{tt}\n")
    return out


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Discrete SIR runner with NetRate per-agent output")
    ap.add_argument("-r", "--run-config", required=True, help="Path to INI (e.g., ../config/verona_sir.ini)")
    ap.add_argument("name", help="Experiment name tag (e.g., netrate_agent_test)")
    args = ap.parse_args()

    cfg_path = Path(args.run_config)
    st = load_settings(cfg_path)

    # Seeds
    random.seed(st.seed)
    np.random.seed(st.seed)

    # Load graph once
    G = load_graph(st.nodes_csv, st.edges_csv)

    print(f"[GRAPH] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    if G.number_of_edges() == 0:
        # show a hint to debug
        some = list(G.nodes())[:5]
        print("[GRAPH] sample node ids:", some)

    degs = np.array([d for _, d in G.degree()])
    print(f"[GRAPH] avg degree={degs.mean():.2f}, isolates={np.sum(degs == 0)}")

    # Parameter grid over betas × I_durations
    for beta in st.betas:
        for I_dur in st.I_durations:
            print(f"\n=== Running beta={beta}, I_duration={I_dur} ===")

            state, history_rows = simulate_sir(
                G,
                beta=beta,
                I_duration=I_dur,
                init_I=st.init_I,
                duration=st.duration,
                seed=st.seed,
                verbose=st.verbose,
                print_interval=st.print_interval,
            )

            # Compose IDs mirroring your earlier files
            out_id = f"_MODEL_beta={beta}_MODEL_I_duration={I_dur}"
            tag = f"history_{args.name}{out_id}"

            header_meta = {
                "edges": str(st.edges_csv),
                "nodes": str(st.nodes_csv),
                "type": "light",
                "I_duration": str(I_dur),
                "beta": str(beta),
                "init_I": str(st.init_I),
            }

            zip_path = write_history_zip(st.out_dir, tag, header_meta, history_rows)
            netrate_path = write_netrate_csv(st.out_dir, out_id, st.seed, state.infection_time)

            print(f"[OK] History written: {zip_path}")
            print(f"[NetRate] wrote per-agent cascade: {netrate_path}")


if __name__ == "__main__":
    main()
