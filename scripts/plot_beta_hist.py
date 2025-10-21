import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
p = Path(__file__).resolve().parent.parent / "data" / "output" / "model" / "netrate_result.csv"
df = pd.read_csv(p)
plt.hist(df["beta"], bins=40)
plt.xlabel("β (transmission rate)")
plt.ylabel("Count of edges")
plt.title("Distribution of inferred transmission rates")
plt.show()