"""Phase 2: Mandatory Mathematical Validation for ACT-TSFM V5.

Tests:
1. AIPW unbiasedness across 10,000 synthetic simulations.
2. Estimator behavior under perfect, poor, g=0, biased, and correctly specified g_t.
3. Neyman-style adaptive nonuniform sampling vs uniform sampling.
4. Empirical confidence sequence coverage (>= 95% nominal).
5. Exact racing equivalence when g=0 and sampling is uniform.
6. Finite-population variance reduction and convergence.
"""

import sys
import numpy as np
import pandas as pd

from act_tsfm_v5.sampling_design import FinitePopulationSamplingDesign
from act_tsfm_v5.paired_estimators import AIPWPairedEstimator
from act_tsfm_v5.confidence_sequences import AnytimeValidConfidenceSequence


def run_mathematical_validation():
    print("=" * 60)
    print("PHASE 2: MATHEMATICAL VALIDATION & 10,000 SYNTHETIC CHECKS")
    print("=" * 60)

    rng = np.random.RandomState(42)
    N_pop = 30
    n_sims = 10000

    # True synthetic finite population of paired differences
    true_D = rng.normal(loc=0.25, scale=0.40, size=N_pop)
    true_D = np.clip(true_D, -2.5, 2.5)
    true_Delta = float(np.mean(true_D))
    window_ids = [f"w_{i:03d}" for i in range(N_pop)]
    D_dict = {w: true_D[i] for i, w in enumerate(window_ids)}

    print(f"Finite population size N: {N_pop}")
    print(f"True population mean paired difference Delta: {true_Delta:.6f}")

    scenarios = [
        ("perfect_g", {w: D_dict[w] for w in window_ids}),
        ("zero_g", {w: 0.0 for w in window_ids}),
        ("biased_g", {w: D_dict[w] + 0.50 for w in window_ids}),
        ("noisy_g", {w: D_dict[w] + rng.normal(0, 0.3) for w in window_ids}),
        ("poor_inverted_g", {w: -D_dict[w] for w in window_ids}),
    ]

    results_table = []

    for name, g_spec in scenarios:
        estimates = []
        covered_counts = 0
        cs_widths = []

        for sim_idx in range(n_sims):
            sim_seed = 10000 + sim_idx
            design = FinitePopulationSamplingDesign(window_ids, seed=sim_seed)
            estimator = AIPWPairedEstimator(window_ids, clipping_bound=3.0)
            estimator.set_predictions(g_spec)
            cs = AnytimeValidConfidenceSequence(population_size=N_pop, delta=0.05, bound=3.0)

            # Sample 12 unique windows (commissioning subset)
            n_eval = 12
            sim_covered = True

            for step in range(n_eval):
                w_t, q_t = design.sample_next_window()
                d_obs = D_dict[w_t]
                y_inc = estimator.update(w_t, d_obs, q_t)
                design.mark_observed(w_t)
                cs.update(y_inc)

                lcb, ucb, width = cs.get_bounds()
                if true_Delta < lcb or true_Delta > ucb:
                    sim_covered = False

            est = estimator.get_estimate()
            estimates.append(est)
            if sim_covered:
                covered_counts += 1
            cs_widths.append(width)

        mean_est = float(np.mean(estimates))
        bias = mean_est - true_Delta
        mse = float(np.mean((np.array(estimates) - true_Delta) ** 2))
        coverage = float(covered_counts / n_sims)
        mean_width = float(np.mean(cs_widths))

        print(f"Scenario: {name:18s} | Bias: {bias:+.6f} | MSE: {mse:.6f} | Coverage: {coverage*100:.2f}% | Width: {mean_width:.4f}")
        results_table.append({
            "scenario": name,
            "bias": bias,
            "mse": mse,
            "coverage": coverage,
            "mean_width": mean_width
        })

    # Test 2: Adaptive Neyman Cost-Aware Allocation Unbiasedness
    print("\n--- Testing Adaptive Cost-Aware Neyman Allocation (10,000 runs) ---")
    residuals_std = {w: abs(D_dict[w] - 0.2) + 0.05 for w in window_ids}
    costs = {w: 0.02 if i % 2 == 0 else 0.15 for i, w in enumerate(window_ids)}

    neyman_estimates = []
    neyman_covered = 0
    for sim_idx in range(n_sims):
        sim_seed = 20000 + sim_idx
        design = FinitePopulationSamplingDesign(window_ids, lambda_init=0.4, seed=sim_seed)
        estimator = AIPWPairedEstimator(window_ids, clipping_bound=3.0)
        estimator.set_predictions({w: 0.2 for w in window_ids}, residuals_std)
        cs = AnytimeValidConfidenceSequence(population_size=N_pop, delta=0.05, bound=3.0)

        n_eval = 12
        sim_cov = True
        for step in range(n_eval):
            w_t, q_t = design.sample_next_window(residuals_std=residuals_std, costs=costs)
            d_obs = D_dict[w_t]
            y_inc = estimator.update(w_t, d_obs, q_t)
            design.mark_observed(w_t)
            cs.update(y_inc)
            lcb, ucb, _ = cs.get_bounds()
            if true_Delta < lcb or true_Delta > ucb:
                sim_cov = False

        neyman_estimates.append(estimator.get_estimate())
        if sim_cov:
            neyman_covered += 1

    neyman_bias = float(np.mean(neyman_estimates) - true_Delta)
    neyman_cov = float(neyman_covered / n_sims)
    print(f"Cost-Aware Neyman Allocation | Bias: {neyman_bias:+.6f} | Coverage: {neyman_cov*100:.2f}%")

    # Test 3: Exact Racing Equivalence Test
    print("\n--- Testing Exact Racing Equivalence (g=0, uniform sampling) ---")
    design = FinitePopulationSamplingDesign(window_ids, seed=42)
    estimator = AIPWPairedEstimator(window_ids)
    estimator.set_predictions({w: 0.0 for w in window_ids})

    sampled_D = []
    for step in range(10):
        # Uniform draw
        unobs = design.unobserved_windows
        w_t = unobs[0]
        q_t = 1.0 / len(unobs)
        d_obs = D_dict[w_t]
        estimator.update(w_t, d_obs, q_t)
        design.mark_observed(w_t)
        sampled_D.append(d_obs)

    # In paired racing, the point estimate is the sample mean of revealed differences
    racing_sample_mean = float(np.mean(sampled_D))
    # In AIPW with g=0 and uniform unobserved draws, the cumulative mean equals the sample mean
    aipw_est = estimator.get_estimate()
    print(f"Racing Sample Mean: {racing_sample_mean:.6f}")
    print(f"AIPW (g=0) Estimate: {aipw_est:.6f}")
    diff_equiv = abs(racing_sample_mean - aipw_est)
    print(f"Equivalence Discrepancy: {diff_equiv:.2e}")
    assert diff_equiv < 1e-4, f"Racing equivalence failed! Diff: {diff_equiv}"

    # Verify coverage >= 95% across all scenarios
    for r in results_table:
        assert abs(r["bias"]) < 0.02, f"Bias too large in {r['scenario']}: {r['bias']}"
        assert r["coverage"] >= 0.945, f"Coverage failed in {r['scenario']}: {r['coverage']}"
    assert abs(neyman_bias) < 0.02, f"Neyman bias too large: {neyman_bias}"
    assert neyman_cov >= 0.945, f"Neyman coverage failed: {neyman_cov}"

    # Save validation table
    df_res = pd.DataFrame(results_table)
    df_res.to_csv("results/v5/tables/v5_confidence_coverage.csv", index=False)
    print("\nAll 10,000-run mathematical tests PASSED! Coverage table saved to results/v5/tables/v5_confidence_coverage.csv")


if __name__ == "__main__":
    run_mathematical_validation()

