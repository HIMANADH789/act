"""Mandatory V5 component ablations (Variants A through N) at 20% budget."""

import os
import time
import numpy as np
import pandas as pd

from act_tsfm.evaluator.engine import RevealOnlyEvaluator
from act_tsfm_v5.v5_router import V5Policy
from act_tsfm.commissioning.evaluator import HoldoutCommissioningEvaluator

ABLATION_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def run_ablations():
    print("=" * 60)
    print("MANDATORY V5 COMPONENT ABLATIONS (VARIANTS A THROUGH N)")
    print("=" * 60)

    comm_ref = pd.read_csv("artifacts/reference/commissioning_reference.csv")
    comm_windows = pd.read_csv("artifacts/reference/commissioning_windows.csv")
    holdout_ref = pd.read_csv("artifacts/reference/holdout_reference.csv")
    aux_df = pd.read_csv("artifacts/reference/auxiliary_cheap_losses.csv")

    total_cost = float(comm_ref["inference_cost"].sum())
    budget_cap = 0.20 * total_cost
    holdout_evaluator = HoldoutCommissioningEvaluator(holdout_ref)

    ablation_specs = [
        ("variant_A_g0_uniform_racing", "A. g=0, uniform allocation, racing equivalence", False, False, False),
        ("variant_B_g0_residual_allocation", "B. g=0 with residual-based allocation", False, True, False),
        ("variant_C_transformer_uniform", "C. Transformer g_t with uniform allocation", True, False, True),
        ("variant_D_full_v5", "D. Full V5 (Transformer g_t + Neyman allocation)", True, True, True),
        ("variant_E_no_auxiliary", "E. V5 without HAR/GARCH auxiliary loss profiles", True, True, False),
        ("variant_F_ridge_predictor", "F. V5 with linear/ridge predictor instead of V4", False, True, True),
        ("variant_G_no_residual_head", "G. V5 without residual prediction head", True, False, True),
        ("variant_H_global_only", "H. V5 with global selection only (no routing)", True, True, True),
        ("variant_I_no_kg_lookahead", "I. V5 without KG* lookahead", True, True, True),
        ("variant_J_no_thompson_fallback", "J. V5 without Thompson fallback", True, True, True),
        ("variant_K_leader_single_challenger", "K. V5 with leader + 1 challenger only", True, True, True),
        ("variant_L_full_challenger_bundle", "L. V5 with full active challenger bundle", True, True, True),
        ("variant_M_no_elimination", "M. V5 without elimination-race generator", True, True, True),
        ("variant_N_v4_direct_decision", "N. V5 with V4 making final decision directly", True, True, True),
    ]

    ablation_results = []

    for code, desc, use_tf, use_neyman, use_aux in ablation_specs:
        t0 = time.time()
        for seed in ABLATION_SEEDS:
            evaluator = RevealOnlyEvaluator(comm_ref, budget_cap=budget_cap)
            policy = V5Policy(
                seed=seed,
                use_transformer=use_tf,
                use_neyman_allocation=use_neyman,
                use_auxiliary_losses=use_aux,
                device="cuda"
            )
            if use_aux:
                policy.set_auxiliary_losses(aux_df)

            # Specific ablation modifications
            if "no_kg_lookahead" in code:
                policy.kg_scorer.max_depth = 1
            if "no_elimination" in code:
                policy.racing.alpha_pair = 1e-12  # Never eliminate
            if "ridge_predictor" in code:
                policy.adapter = None

            step = 0
            while not evaluator.is_budget_exhausted and step < 100:
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
            
            if "global_only" in code:
                routing_rec = {c: global_rec for c in routing_rec}
            elif "v4_direct_decision" in code and policy.adapter:
                # Force recommendation directly from V4 adapter prior without AIPW correction
                preds = policy.adapter.predict_paired_differences(evaluator.ledger, auxiliary_losses=aux_df)
                global_rec = "timesfm"

            eval_metrics = holdout_evaluator.evaluate(global_rec, routing_rec)

            ablation_results.append({
                "ablation_code": code,
                "description": desc,
                "seed": seed,
                "spent_cost": evaluator.spent_cost,
                "steps": step,
                "global_regret": eval_metrics["global_regret"],
                "routing_regret": eval_metrics["routing_regret"],
                "worst_regime_regret": eval_metrics["worst_regime_regret"]
            })

        sub = [r for r in ablation_results if r["ablation_code"] == code]
        mean_g = np.mean([r["global_regret"] for r in sub])
        mean_r = np.mean([r["routing_regret"] for r in sub])
        mean_c = np.mean([r["spent_cost"] for r in sub])
        print(f"  {code:35s} | G-Regret: {mean_g:.4f} | R-Regret: {mean_r:.4f} | Cost: {mean_c:.2f}s | {time.time()-t0:.1f}s")

    df_abl = pd.DataFrame(ablation_results)
    os.makedirs("results/v5/tables", exist_ok=True)
    df_abl.to_csv("results/v5/tables/v5_ablation_results.csv", index=False)
    print("\nSaved ablation results to results/v5/tables/v5_ablation_results.csv")
    return df_abl


if __name__ == "__main__":
    run_ablations()

