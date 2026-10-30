"""Phase 6: Expanded 32-model candidate pool scalability stress benchmark."""

import os
import time
import numpy as np
import pandas as pd

from act_tsfm.evaluator.engine import RevealOnlyEvaluator
from act_tsfm_v5.v5_router import V5Policy
from act_tsfm.commissioning.evaluator import HoldoutCommissioningEvaluator

EXPANDED_MODELS = [
    "naive", "hist_mean", "ewma_05", "ewma_10", "ewma_20",
    "ar1", "ar5", "ar22",
    "har_base", "har_log", "har_q", "har_rolling",
    "garch_normal", "garch_t", "garch_skewt", "gjr_garch", "egarch", "garch_rolling",
    "ridge_rv", "lasso_rv", "enet_rv",
    "rf_50", "rf_100", "gbr_rv",
    "mlp_small", "mlp_medium",
    "chronos_tiny", "chronos_sample",
    "timesfm_base", "timesfm_sample",
    "ensemble_equal", "ensemble_inv_var"
]

BUDGET_FRACTIONS = [0.10, 0.20, 0.40, 0.60, 0.80]
EXP_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def run_expanded_benchmark():
    print("=" * 60)
    print("PHASE 6: EXPANDED 32-MODEL SCALABILITY STRESS BENCHMARK")
    print("=" * 60)

    comm_ref = pd.read_csv("artifacts/reference/expanded_commissioning_reference.csv")
    comm_windows = pd.read_csv("artifacts/reference/commissioning_windows.csv")
    holdout_ref = pd.read_csv("artifacts/reference/expanded_holdout_reference.csv")
    aux_df = pd.read_csv("artifacts/reference/auxiliary_cheap_losses.csv")

    total_cost = float(comm_ref["inference_cost"].sum())
    holdout_evaluator = HoldoutCommissioningEvaluator(holdout_ref)

    print(f"Total potential commissioning cost across 32 models: {total_cost:.3f}s")
    print(f"Holdout Global Oracle: {holdout_evaluator.oracle_global_model} (Loss: {holdout_evaluator.oracle_global_loss:.4f})")
    print(f"Holdout Routing Oracle Loss: {holdout_evaluator.oracle_routing_loss:.4f}")

    results = []

    for b_frac in BUDGET_FRACTIONS:
        budget_cap = b_frac * total_cost
        print(f"\n--- Budget {b_frac*100:.0f}% (Cap: {budget_cap:.2f}s) ---")

        for m_name in ["v5", "racing"]:
            t0 = time.time()
            for seed in EXP_SEEDS:
                evaluator = RevealOnlyEvaluator(comm_ref, budget_cap=budget_cap)

                if m_name == "v5":
                    policy = V5Policy(
                        seed=seed,
                        models=EXPANDED_MODELS,
                        use_transformer=True,
                        use_neyman_allocation=True,
                        use_auxiliary_losses=True,
                        device="cuda"
                    )
                    policy.set_auxiliary_losses(aux_df)
                else:
                    # Paired racing baseline on 32 models: g=0, uniform sampling
                    policy = V5Policy(
                        seed=seed,
                        models=EXPANDED_MODELS,
                        use_transformer=False,
                        use_neyman_allocation=False,
                        use_auxiliary_losses=False,
                        device="cpu"
                    )

                step = 0
                while not evaluator.is_budget_exhausted and step < 120:
                    prev_spent = evaluator.spent_cost
                    action = policy.select_action(evaluator, comm_windows)
                    if action is None:
                        break
                    models, w_id = action
                    evaluator.evaluate_bundle(models, w_id, step_idx=step)
                    if evaluator.spent_cost <= prev_spent + 1e-9:
                        break
                    step += 1

                global_rec, routing_rec = policy.make_recommendations(evaluator)
                eval_metrics = holdout_evaluator.evaluate(global_rec, routing_rec)

                results.append({
                    "budget_fraction": b_frac,
                    "budget_cap": budget_cap,
                    "method": m_name,
                    "seed": seed,
                    "spent_cost": evaluator.spent_cost,
                    "n_evaluations": len(evaluator.ledger),
                    "global_recommendation": global_rec,
                    "global_regret": eval_metrics["global_regret"],
                    "routing_regret": eval_metrics["routing_regret"]
                })

            sub = [r for r in results if r["budget_fraction"] == b_frac and r["method"] == m_name]
            mean_g = np.mean([r["global_regret"] for r in sub])
            mean_r = np.mean([r["routing_regret"] for r in sub])
            mean_c = np.mean([r["spent_cost"] for r in sub])
            print(f"  {m_name:10s} | G-Regret: {mean_g:.4f} | R-Regret: {mean_r:.4f} | Cost: {mean_c:.2f}s | {time.time()-t0:.1f}s")

    df_exp = pd.DataFrame(results)
    os.makedirs("results/v5/tables", exist_ok=True)
    df_exp.to_csv("results/v5/tables/v5_expanded_pool_results.csv", index=False)
    print("\nSaved expanded benchmark results to results/v5/tables/v5_expanded_pool_results.csv")
    return df_exp


if __name__ == "__main__":
    run_expanded_benchmark()

