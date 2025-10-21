# scripts/peek_cascades.py
from pathlib import Path
import pandas as pd
p = Path(__file__).resolve().parents[2] / "data" / "output" / "model" / "netrate" / "netrate_cascades_all.csv"
df = pd.read_csv(p)
print(df.head(10).to_string(index=False))
print("\nCascades:", df["cascade_id"].nunique(), " Unique nodes:", df["node_id"].nunique())
print(df.groupby("cascade_id")["node_id"].nunique().describe())
