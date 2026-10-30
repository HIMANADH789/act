"""Phase 4: Smoke experiment for ACT-TSFM V5."""

import pandas as pd
from act_tsfm.evaluator.engine import RevealOnlyEvaluator
from act_tsfm_v5.v5_router import V5Policy
from act_tsfm.commissioning.evaluator import HoldoutCommissioningEvaluator


def run_smoke():
    print("=" * 60)
    print("PHASE 4: V5 SMOKE EXPERIMENT")
    print("=" * 60)

    comm_ref = pd.read_csv("artifacts/reference/commissioning_reference.csv")
    comm_windows = pd.read_csv("artifacts/reference/commissioning_windows.csv")
    holdout_ref = pd.read_csv("artifacts/reference/holdout_reference.csv")
    aux_df = pd.read_csv("artifacts/reference/auxiliary_cheap_losses.csv")

    holdout_evaluator = HoldoutCommissioningEvaluator(holdout_ref)

    # Test across seeds 42, 43 at 20% budget
    total_cost = float(comm_ref["inference_cost"].sum())
    budget_cap = 0.20 * total_cost

    for seed in [42, 43]:
        print(f"\n--- Running V5 Smoke for Seed {seed} (Cap: {budget_cap:.2f}s) ---")
        evaluator = RevealOnlyEvaluator(comm_ref, budget_cap=budget_cap)
        policy = V5Policy(seed=seed, device="cpu")
        policy.set_auxiliary_losses(aux_df)

        step = 0
        while not evaluator.is_budget_exhausted and step < 50:
            prev_spent = evaluator.spent_cost
            action = policy.select_action(evaluator, comm_windows)
            if action is None:
                print(f"Policy stopped at step {step}")
                break
            models, w_id = action
            evaluator.evaluate_bundle(models, w_id, step_idx=step)
            if evaluator.spent_cost <= prev_spent + 1e-9:
                break
            step += 1

        global_rec, routing_rec = policy.make_recommendations(evaluator)
        cert_cells = policy.certifier.certified_cells
        eval_metrics = holdout_evaluator.evaluate(global_rec, routing_rec)

        print(f"Steps: {step} | Spent Cost: {evaluator.spent_cost:.3f}s / {budget_cap:.2f}s | Ledger Entries: {len(evaluator.ledger)}")
        print(f"Global Recommendation: {global_rec} (Holdout Loss: {eval_metrics['global_loss']:.4f}, Regret: {eval_metrics['global_regret']:.4f})")
        print(f"Routing Regret: {eval_metrics['routing_regret']:.4f}")
        print(f"Certified Cells ({len(cert_cells)}/9): {cert_cells}")

    print("\nV5 SMOKE EXPERIMENT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_smoke()

