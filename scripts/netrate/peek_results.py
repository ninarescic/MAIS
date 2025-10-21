from scripts.netrate.utils_netrate import data_dir, load_results

p = Path(__file__).resolve().parents[2] / "data" / "output" / "model" / "netrate_result_true.csv"
df = load_results(data_dir("netrate_result.csv"))
print("Edges:", len(df))
print("\nTop 20 by beta:")
print(df.sort_values("beta", ascending=False).head(20).to_string(index=False))