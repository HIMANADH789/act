"""Builds the full evaluator reference matrix on Commissioning and Holdout periods."""

import os
import json
import time
import pandas as pd
import numpy as np

from act_tsfm.volare_data import (
    load_raw_volare, 
    process_features_and_targets, 
    extract_evaluation_windows
)
from act_tsfm.forecasting.adapters import build_forecasters, qlike_loss

def main():
    print("=" * 60)
    print("BUILDING FULL REFERENCE EVALUATION MATRICES (VOLARE)")
    print("=" * 60)
    
    os.makedirs("artifacts/reference", exist_ok=True)
    os.makedirs("results/manifests", exist_ok=True)
    
    raw = load_raw_volare()
    proc, thresholds = process_features_and_targets(raw)
    
    # Save regime thresholds
    with open("results/manifests/regime_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Regime thresholds saved:", thresholds)
    
    # Extract windows
    comm_windows = extract_evaluation_windows(proc, "commissioning", windows_per_asset_cell=4, seed=42)
    holdout_windows = extract_evaluation_windows(proc, "holdout", windows_per_asset_cell=2, seed=42)
    
    print(f"Commissioning windows: {len(comm_windows)}")
    print(f"Holdout windows:       {len(holdout_windows)}")
    
    comm_windows.to_csv("artifacts/reference/commissioning_windows.csv", index=False)
    holdout_windows.to_csv("artifacts/reference/holdout_windows.csv", index=False)
    
    # Fit models on development data
    train_histories = {sym: df[df["partition"] == "development"] for sym, df in proc.items()}
    print("\nFitting forecasters on Development partition (through 2023-12-31)...")
    forecasters = build_forecasters(device="cuda", seed=42)
    for name, f in forecasters.items():
        f.fit(train_histories)
        print(f"  {name:8s} fit time: {f.fit_time_seconds:.3f}s")
        
    for split_name, windows in [("commissioning", comm_windows), ("holdout", holdout_windows)]:
        print(f"\nEvaluating forecasters on {split_name.upper()} period ({len(windows)} windows)...")
        rows = []
        t0_split = time.time()
        
        for idx, w in windows.iterrows():
            sym = w["asset_id"]
            origin = w["forecast_origin"]
            h = int(w["horizon"])
            actual = float(w["target_variance"])
            history_df = proc[sym]
            
            for m_id, f in forecasters.items():
                pred, inf_sec = f.predict(sym, history_df, origin, h)
                loss = qlike_loss(actual, pred)
                
                rows.append({
                    "model_id": m_id,
                    "model_version": f.model_version,
                    "window_id": w["window_id"],
                    "asset_id": sym,
                    "category": w["category"],
                    "forecast_origin": origin,
                    "target_end_date": w["target_end_date"],
                    "horizon": h,
                    "regime": w["regime"],
                    "cell_id": w["cell_id"],
                    "predicted_variance": pred,
                    "actual_variance": actual,
                    "qlike_loss": loss,
                    "inference_cost": max(inf_sec, 0.001),
                    "fit_time_seconds": f.fit_time_seconds,
                    "evaluation_seed": 42
                })
            if (idx + 1) % 50 == 0 or (idx + 1) == len(windows):
                print(f"  Processed {idx + 1}/{len(windows)} windows in {time.time() - t0_split:.1f}s")
                
        ref_df = pd.DataFrame(rows)
        out_csv = f"artifacts/reference/{split_name}_reference.csv"
        ref_df.to_csv(out_csv, index=False)
        print(f"Saved {len(ref_df)} {split_name} records to {out_csv}")
        
    print("\nReference matrix construction complete!")

if __name__ == "__main__":
    main()

