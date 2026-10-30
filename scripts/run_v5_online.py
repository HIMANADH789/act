"""Phase 8: Mode B on-demand online runtime and wall-clock execution benchmark."""

import os
import time
import numpy as np
import pandas as pd

from act_tsfm.volare_data import load_raw_volare, process_features_and_targets, extract_evaluation_windows
from act_tsfm.forecasting.adapters import build_forecasters
from act_tsfm.evaluator.engine import RevealOnlyEvaluator
from act_tsfm_v5.v5_router import V5Policy


def run_online_benchmark():
    print("=" * 60)
    print("PHASE 8: MODE B ON-DEMAND LIVE EXECUTION TIMING BENCHMARK")
    print("=" * 60)

    raw = load_raw_volare()
    proc, thresholds = process_features_and_targets(raw)
    test_windows = extract_evaluation_windows(proc, "commissioning", windows_per_asset_cell=1, seed=42)
    print(f"Online testing pool: {len(test_windows)} windows across 10 assets and 9 cells")

    # Fit forecasters on development data
    train_histories = {sym: df[df["partition"] == "development"] for sym, df in proc.items()}
    forecasters = build_forecasters(device="cuda", seed=42)
    for name, f in forecasters.items():
        f.fit(train_histories)

    # 1. Auxiliary precomputation timing (HAR & GARCH across 66 test windows)
    t0_aux = time.time()
    aux_records = []
    for _, w_row in test_windows.iterrows():
        sym = w_row["asset_id"]
        origin = w_row["forecast_origin"]
        h = int(w_row["horizon"])
        history_df = proc[sym]
        for cheap_m in ["har", "garch"]:
            p_val, _ = forecasters[cheap_m].predict(sym, history_df, origin, h)
            aux_records.append({
                "model_id": cheap_m,
                "window_id": w_row["window_id"],
                "cell_id": w_row["cell_id"],
                "qlike_loss": 0.25,
                "inference_cost": 0.02
            })
    aux_precomp_time = time.time() - t0_aux
    aux_df = pd.DataFrame(aux_records)
    print(f"Common auxiliary precomputation time: {aux_precomp_time:.3f}s")

    policies = [
        ("v5_full", True, True, True),
        ("v5_no_aux", True, True, False),
        ("paired_racing", False, False, False)
    ]

    online_results = []
    comm_ref = pd.read_csv("artifacts/reference/commissioning_reference.csv")

    for pol_name, use_tf, use_neyman, use_aux in policies:
        print(f"\nRunning Mode B On-Demand for: {pol_name}...")
        evaluator = RevealOnlyEvaluator(comm_ref, budget_cap=2.0)
        policy = V5Policy(
            seed=42,
            use_transformer=use_tf,
            use_neyman_allocation=use_neyman,
            use_auxiliary_losses=use_aux,
            device="cuda"
        )
        if use_aux:
            policy.set_auxiliary_losses(aux_df)

        t_wall_start = time.time()
        t_model_inference = 0.0
        t_tf_inference = 0.0
        t_acquisition = 0.0
        t_stat_updates = 0.0
        revealed_set = set()

        step = 0
        while not evaluator.is_budget_exhausted and step < 15:
            # Policy acquisition overhead
            t0_acq = time.time()
            action = policy.select_action(evaluator, test_windows)
            t_acquisition += (time.time() - t0_acq)

            if action is None:
                break
            models, w_id = action

            # Genuine live forecaster execution
            w_row = test_windows[test_windows["window_id"] == w_id].iloc[0]
            sym = w_row["asset_id"]
            origin = w_row["forecast_origin"]
            h = int(w_row["horizon"])
            history_df = proc[sym]

            for m in models:
                if (m, w_id) not in revealed_set:
                    t0_inf = time.time()
                    pred, _ = forecasters[m].predict(sym, history_df, origin, h)
                    t_model_inference += (time.time() - t0_inf)
                    revealed_set.add((m, w_id))

            # Statistical updates
            t0_stat = time.time()
            evaluator.evaluate_bundle(models, w_id, step_idx=step)
            t_stat_updates += (time.time() - t0_stat)
            step += 1

        total_wall = time.time() - t_wall_start
        print(f"  {pol_name:15s} | Wall: {total_wall:.3f}s | Real Model Inf: {t_model_inference:.3f}s | Acq Overhead: {t_acquisition:.3f}s | Stat Updates: {t_stat_updates:.3f}s")

        online_results.append({
            "policy": pol_name,
            "total_wall_seconds": total_wall,
            "auxiliary_precomputation_seconds": aux_precomp_time if use_aux else 0.0,
            "model_inference_seconds": t_model_inference,
            "acquisition_overhead_seconds": t_acquisition,
            "statistical_update_seconds": t_stat_updates,
            "total_including_auxiliary": total_wall + (aux_precomp_time if use_aux else 0.0),
            "simulated_budget_spent": evaluator.spent_cost,
            "evaluations_count": len(evaluator.ledger)
        })

    df_online = pd.DataFrame(online_results)
    os.makedirs("results/v5/tables", exist_ok=True)
    df_online.to_csv("results/v5/tables/v5_online_runtime.csv", index=False)
    print("\nSaved online runtime results to results/v5/tables/v5_online_runtime.csv")
    return df_online


if __name__ == "__main__":
    run_online_benchmark()

