"""Mode B: On-Demand Online Execution Check comparing genuine inference and policy overhead."""

import os
import time
import pandas as pd
import numpy as np

from act_tsfm.volare_data import load_raw_volare, process_features_and_targets, extract_evaluation_windows
from act_tsfm.forecasting.adapters import build_forecasters, qlike_loss
from act_tsfm.evaluator.engine import RevealOnlyEvaluator
from act_tsfm.baselines.selection_strategies import get_baseline_policy
from act_tsfm.v1.bayesian_bilinear import V1Policy
from act_tsfm.v3.paired_router import V3Policy
from act_tsfm.v4.axial_transformer import V4Policy

def main():
    print("=" * 60)
    print("MODE B: ON-DEMAND GENUINE EXECUTION TIMING CHECK")
    print("=" * 60)
    
    os.makedirs("results/tables", exist_ok=True)
    raw = load_raw_volare()
    proc, thresholds = process_features_and_targets(raw)
    
    # 1 small pool: 1 window per asset/cell
    test_windows = extract_evaluation_windows(proc, "commissioning", windows_per_asset_cell=1, seed=42)
    print(f"Online testing pool: {len(test_windows)} windows across 10 assets and 9 cells")
    
    # Fit forecasters
    train_histories = {sym: df[df["partition"] == "development"] for sym, df in proc.items()}
    forecasters = build_forecasters(device="cuda", seed=42)
    for name, f in forecasters.items():
        f.fit(train_histories)
        
    policies_to_test = [
        ("fixed", get_baseline_policy("fixed", seed=42)),
        ("random", get_baseline_policy("random", seed=42)),
        ("racing", get_baseline_policy("racing", seed=42)),
        ("v1", V1Policy(seed=42)),
        ("v3", V3Policy(seed=42)),
        ("v4", V4Policy(seed=42, device="cuda")),
    ]
    
    online_results = []
    
    for pol_name, policy in policies_to_test:
        print(f"\nRunning Mode B On-Demand for: {pol_name.upper()}...")
        wall_start = time.time()
        inference_time_total = 0.0
        policy_overhead_total = 0.0
        eval_calls = 0
        
        # In-memory tracking of revealed observations
        revealed = set()
        sim_evaluator = RevealOnlyEvaluator(pd.read_csv("artifacts/reference/commissioning_reference.csv"), budget_cap=2.0)
        
        step = 0
        while not sim_evaluator.is_budget_exhausted and step < 20:
            t0_pol = time.time()
            action = policy.select_action(sim_evaluator, test_windows)
            policy_overhead_total += (time.time() - t0_pol)
            
            if action is None:
                break
            models, w_id = action
            
            # Genuine on-demand execution call
            w_row = test_windows[test_windows["window_id"] == w_id].iloc[0]
            sym = w_row["asset_id"]
            origin = w_row["forecast_origin"]
            h = int(w_row["horizon"])
            history_df = proc[sym]
            
            for m in models:
                if (m, w_id) not in revealed:
                    t0_inf = time.time()
                    pred, _ = forecasters[m].predict(sym, history_df, origin, h)
                    inference_time_total += (time.time() - t0_inf)
                    revealed.add((m, w_id))
                    eval_calls += 1
                    
            sim_evaluator.evaluate_bundle(models, w_id, step_idx=step)
            step += 1
            
        total_wall = time.time() - wall_start
        print(f"  {pol_name:8s} | Wall: {total_wall:.2f}s | Real Inf: {inference_time_total:.2f}s | Policy Overhead: {policy_overhead_total:.2f}s | Calls: {eval_calls}")
        
        online_results.append({
            "method": pol_name,
            "wall_seconds": total_wall,
            "inference_seconds": inference_time_total,
            "policy_overhead_seconds": policy_overhead_total,
            "evaluation_calls": eval_calls,
            "simulated_budget_spent": sim_evaluator.spent_cost
        })
        
    df_online = pd.DataFrame(online_results)
    df_online.to_csv("results/tables/online_execution_results.csv", index=False)
    print("\nSaved Mode B online execution results to results/tables/online_execution_results.csv")

if __name__ == "__main__":
    main()

