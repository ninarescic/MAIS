"""
Batch runner to generate multiple NetRate-compatible cascades
from the Verona SIR model using MAIS simulation.
"""

import subprocess
import re
import pandas as pd
from pathlib import Path

# === Paths ===
ROOT = Path(__file__).resolve().parent.parent
INI = ROOT / "config" / "verona_sir.ini"
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "data" / "output" / "model" / "netrate"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Seeds ===
# Each seed produces a slightly different cascade (different random infection order)
seeds = list(range(1, 31))  # 30 runs

print(f"[Batch] Running {len(seeds)} seeds...")

ini_text = INI.read_text(encoding="utf-8")

def write_ini(seed):
    """Create a temporary INI for a specific seed — replacing, not duplicating."""
    txt = ini_text

    # Replace random_seed if it exists anywhere
    if re.search(r"(?im)^\s*random_seed\s*=", txt):
        txt = re.sub(r"(?im)^(random_seed\s*=\s*)\d+", rf"\1{seed}", txt)
    else:
        # Find [TASK] section; insert random_seed line under it
        if "[TASK]" in txt:
            txt = re.sub(r"(?i)(\[TASK\])", rf"\1\nrandom_seed = {seed}", txt)
        else:
            # If there's no [TASK], add it at the end
            txt += f"\n[TASK]\nrandom_seed = {seed}\n"

    new_ini = INI.with_name(f"verona_sir_seed{seed}.ini")
    new_ini.write_text(txt, encoding="utf-8")
    return new_ini


# === Run the simulations ===
for s in seeds:
    ini_path = write_ini(s)
    print(f"[Batch] Seed={s} -> {ini_path.name}")
    subprocess.check_call([
        "python",
        str(SCRIPTS / "run_experiment_netrate.py"),
        "-r",
        str(ini_path),
        "netrate_agent_test",
    ])

# === Combine results ===
cascades = sorted(OUT_DIR.glob("netrate_cascade_*.csv"))
if not cascades:
    raise SystemExit("No cascade CSVs found — check if runs produced output files.")

parts = [pd.read_csv(p) for p in cascades]
all_df = pd.concat(parts, ignore_index=True)
out_file = OUT_DIR / "netrate_cascades_all.csv"
all_df.to_csv(out_file, index=False)
print(f"[Batch] Wrote {out_file} with {len(all_df)} rows from {len(cascades)} cascades.")
