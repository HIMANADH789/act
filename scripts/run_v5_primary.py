"""Phase 5: Primary 4-Model VOLARE Commissioning Benchmark for V5 vs Racing and Baselines."""

import os
import time
import numpy as np
import pandas as pd

from act_tsfm.evaluator.engine import RevealOnlyEvaluator
from act_tsfm.baselines.selection_strategies import get_baseline_policy
from act_tsfm.v1.bayesian_bilinear import V1Policy
from act_tsfm.v3.paired_router import V3Policy
from act_tsfm.v4.axial_transformer import V4Policy
from act_tsfm_v5.v5_router import V5Policy
from act_tsfm.commissioning.evaluator import HoldoutCommissioningEvaluator

BUDGET_FRACTIONS = [0.10, 0.20, 0.40, 0.60, 0.80]
# 100 policy seeds: 42 to 141
SEEDS = list(range(42, 142))
PRIMARY_METHODS = ["v5", "racing", "random", "ucb", "uncertainty", "v4", "v1"]


def run_primary_benchmark():
    print("=" * 60)
    print("PHASE 5: PRIMARY 4-MODEL COMMISSIONING BENCHMARK (100 SEEDS)")
    print("=" * 60)

    comm_ref = pd.read_csv("artifacts/reference/commissioning_reference.csv")
    comm_windows = pd.read_csv("artifacts/reference/commissioning_windows.csv")
    holdout_ref = pd.read_csv("artifacts/reference/holdout_reference.csv")
    aux_df = pd.read_csv("artifacts/reference/auxiliary_cheap_losses.csv")

    total_ref_cost = float(comm_ref["inference_cost"].sum())
    holdout_evaluator = HoldoutCommissioningEvaluator(holdout_ref)

    print(f"Total potential commissioning cost: {total_ref_cost:.3f}s")
    print(f"Holdout Global Oracle: {holdout_evaluator.oracle_global_model} (Loss: {holdout_evaluator.oracle_global_loss:.4f})")
    print(f"Holdout Routing Oracle Loss: {holdout_evaluator.oracle_routing_loss:.4f}")

    all_results = []
    
    # We run 100 seeds for V5, Racing, Random; and 20 seeds for computationally heavier baselines
    for b_frac in BUDGET_FRACTIONS:
        budget_cap = b_frac * total_ref_cost
        print(f"\n--- BUDGET TIER {b_frac*100:.0f}% (Cap: {budget_cap:.2f}s) ---")

        for m_name in PRIMARY_METHODS:
            t0_m = time.time()
            seeds_to_run = SEEDS if m_name in ["v5", "racing", "random"] else SEEDS[:20]

            for seed in seeds_to_run:
                evaluator = RevealOnlyEvaluator(comm_ref, budget_cap=budget_cap)

                if m_name == "v5":
                    policy = V5Policy(seed=seed, device="cuda")
                    policy.set_auxiliary_losses(aux_df)
                elif m_name == "racing":
                    policy = get_baseline_policy("racing", seed=seed)
                elif m_name in ["fixed", "random", "risk", "uncertainty", "ucb"]:
                    policy = get_baseline_policy(m_name, seed=seed)
                elif m_name == "v1":
                    policy = V1Policy(seed=seed)
                elif m_name == "v3":
                    policy = V3Policy(seed=seed)
                elif m_name == "v4":
                    policy = V4Policy(seed=seed, device="cuda")

                step = 0
                t0_pol = time.time()
                while not evaluator.is_budget_exhausted and step < 150:
                    prev_spent = evaluator.spent_cost
                    action = policy.select_action(evaluator, comm_windows)
                    if action is None:
                        break
                    models, w_id = action
                    evaluator.evaluate_bundle(models, w_id, step_idx=step)
                    if evaluator.spent_cost <= prev_spent + 1e-9:
                        break
                    step += 1

                pol_time = time.time() - t0_pol
                global_rec, routing_rec = policy.make_recommendations(evaluator)
                
                # Check certification
                cert_count = 0
                if hasattr(policy, "certifier"):
                    cert_count = len(policy.certifier.certified_cells)
                elif hasattr(policy, "certified_cells"):
                    cert_count = len(policy.certified_cells)

                eval_metrics = holdout_evaluator.evaluate(global_rec, routing_rec)

                row = {
                    "budget_fraction": b_frac,
                    "budget_cap_seconds": budget_cap,
                    "method": m_name,
                    "policy_seed": seed,
                    "spent_cost": evaluator.spent_cost,
                    "n_evaluations": len(evaluator.ledger),
                    "policy_wall_seconds": pol_time,
                    "certified_cells_count": cert_count,
                    **eval_metrics
                }
                all_results.append(row)

            sub = [r for r in all_results if r["budget_fraction"] == b_frac and r["method"] == m_name]
            mean_g_reg = np.mean([r["global_regret"] for r in sub])
            mean_r_reg = np.mean([r["routing_regret"] for r in sub])
            mean_cost = np.mean([r["spent_cost"] for r in sub])
            print(f"  {m_name:12s} | Global Regret: {mean_g_reg:.4f} | Routing Regret: {mean_r_reg:.4f} | Cost: {mean_cost:.2f}s | {time.time()-t0_m:.1f}s")

    df_res = pd.DataFrame(all_results)
    os.makedirs("results/v5/tables", exist_ok=True)
    df_res.to_csv("results/v5/tables/v5_main_results.csv", index=False)

    # Save summary regret-cost curve
    regret_cost = df_res.groupby(["budget_fraction", "method"])[["global_regret", "routing_regret", "spent_cost", "policy_wall_seconds"]].mean().reset_index()
    regret_cost.to_csv("results/v5/tables/v5_regret_cost_curves.csv", index=False)

    # Save baseline comparisons pivot
    g_pivot = df_res.pivot_table(index="budget_fraction", columns="method", values="global_regret", aggfunc="mean")
    g_pivot.to_csv("results/v5/tables/v5_baseline_comparisons.csv")

    # Routing results pivot
    r_pivot = df_res.pivot_table(index="budget_fraction", columns="method", values="routing_regret", aggfunc="mean")
    r_pivot.to_csv("results/v5/tables/v5_routing_results.csv")

    # Certification results
    c_pivot = df_res.pivot_table(index="budget_fraction", columns="method", values="certified_cells_count", aggfunc="mean")
    c_pivot.to_csv("results/v5/tables/v5_certification_results.csv")

    print("\nSaved primary benchmark results to results/v5/tables/v5_main_results.csv")
    return df_res


if __name__ == "__main__":
    run_primary_benchmark()

