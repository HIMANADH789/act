from act_tsfm.volare_data import load_raw_volare, process_features_and_targets, extract_evaluation_windows
from act_tsfm.forecasting.adapters import build_forecasters, qlike_loss

print("Loading VOLARE raw data...")
raw = load_raw_volare()
proc, thresh = process_features_and_targets(raw)

# Get dev histories for fitting
train_histories = {sym: df[df["partition"] == "development"] for sym, df in proc.items()}

print("Fitting forecasters...")
forecasters = build_forecasters(device="cuda", seed=42)

for name in ["har", "garch", "chronos", "timesfm"]:
    f = forecasters[name]
    f.fit(train_histories)
    print(f"  {name} fitted in {f.fit_time_seconds:.3f}s")
    
# Test prediction on AAPL for h=1, h=5, h=22
df_aapl = proc["AAPL"]
origin = "2024-01-02"
target_actual = df_aapl.loc[df_aapl["date"] == origin, "target_h5"].values[0]

print(f"\nPredictions for AAPL on origin {origin} (h=5, actual target={target_actual:.6f}):")
for name in ["har", "garch", "chronos", "timesfm"]:
    f = forecasters[name]
    pred, inf_time = f.predict("AAPL", df_aapl, origin, 5)
    loss = qlike_loss(target_actual, pred)
    print(f"  {name:8s}: pred={pred:.6e}, inf_time={inf_time*1000:.1f}ms, QLIKE={loss:.4f}")

