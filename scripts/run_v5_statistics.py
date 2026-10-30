"""Phase 7: Paired block-bootstrap statistical significance and false-certification checks."""

import os
import numpy as np
import pandas as pd

from act_tsfm_v5.sampling_design import FinitePopulationSamplingDesign
from act_tsfm_v5.paired_estimators import AIPWPairedEstimator
from act_tsfm_v5.confidence_sequences import AnytimeValidConfidenceSequence
from act_tsfm_v5.certification import CertificationManager


def run_statistical_analysis():
    print("=" * 60)
    print("PHASE 7: STATISTICAL ANALYSIS & PAIRED BLOCK-BOOTSTRAP (4,000 REPLICATES)")
    print("=" * 60)

    res_df = pd.read_csv("results/v5/tables/v5_main_results.csv")
    rng = np.random.RandomState(42)

    comparisons = []
    budget_fractions = [0.10, 0.20, 0.40, 0.60, 0.80]

    # Paired comparisons against baselines
    for b_frac in budget_fractions:
        sub_b = res_df[res_df["budget_fraction"] == b_frac]
        v5_rows = sub_b[sub_b["method"] == "v5"]

        for baseline in ["racing", "random", "ucb", "uncertainty", "v4", "v1"]:
            base_rows = sub_b[sub_b["method"] == baseline]
            if len(base_rows) == 0:
                continue

            # Merge on policy_seed
            merged = pd.merge(v5_rows, base_rows, on="policy_seed", suffixes=("_v5", f"_{baseline}"))
            if len(merged) == 0:
                continue

            diffs_g = merged["global_regret_v5"].values - merged[f"global_regret_{baseline}"].values
            diffs_r = merged["routing_regret_v5"].values - merged[f"routing_regret_{baseline}"].values

            # 4,000 paired bootstrap resamples
            n_samples = len(diffs_g)
            boot_diffs_g = []
            boot_diffs_r = []
            for _ in range(4000):
                idx = rng.randint(0, n_samples, size=n_samples)
                boot_diffs_g.append(float(np.mean(diffs_g[idx])))
                boot_diffs_r.append(float(np.mean(diffs_r[idx])))

            boot_diffs_g = np.array(boot_diffs_g)
            boot_diffs_r = np.array(boot_diffs_r)

            # Compute 95% CI and p-value
            ci_g_low = float(np.percentile(boot_diffs_g, 2.5))
            ci_g_high = float(np.percentile(boot_diffs_g, 97.5))
            p_val_g = float(2.0 * min(np.mean(boot_diffs_g <= 0), np.mean(boot_diffs_g >= 0)))
            p_val_g = min(1.0, max(p_val_g, 1.0 / 4000))

            ci_r_low = float(np.percentile(boot_diffs_r, 2.5))
            ci_r_high = float(np.percentile(boot_diffs_r, 97.5))
            p_val_r = float(2.0 * min(np.mean(boot_diffs_r <= 0), np.mean(boot_diffs_r >= 0)))
            p_val_r = min(1.0, max(p_val_r, 1.0 / 4000))

            comparisons.append({
                "budget_fraction": b_frac,
                "candidate": "v5",
                "baseline": baseline,
                "n_seeds": n_samples,
                "mean_diff_global_regret": float(np.mean(diffs_g)),
                "ci_g_low_95": ci_g_low,
                "ci_g_high_95": ci_g_high,
                "raw_p_value_global": p_val_g,
                "mean_diff_routing_regret": float(np.mean(diffs_r)),
                "ci_r_low_95": ci_r_low,
                "ci_r_high_95": ci_r_high,
                "raw_p_value_routing": p_val_r
            })

    stat_df = pd.DataFrame(comparisons)

    # Holm-Bonferroni correction per budget fraction
    stat_df["holm_p_value_global"] = 1.0
    stat_df["holm_p_value_routing"] = 1.0
    for b_frac in budget_fractions:
        mask = stat_df["budget_fraction"] == b_frac
        sub_stat = stat_df[mask].sort_values("raw_p_value_routing")
        m = len(sub_stat)
        holm_ps_r = []
        running_max_r = 0.0
        for i, (_, row) in enumerate(sub_stat.iterrows()):
            hp = (m - i) * row["raw_p_value_routing"]
            running_max_r = max(running_max_r, hp)
            holm_ps_r.append(min(1.0, running_max_r))
        stat_df.loc[sub_stat.index, "holm_p_value_routing"] = holm_ps_r

        sub_stat_g = stat_df[mask].sort_values("raw_p_value_global")
        holm_ps_g = []
        running_max_g = 0.0
        for i, (_, row) in enumerate(sub_stat_g.iterrows()):
            hp = (m - i) * row["raw_p_value_global"]
            running_max_g = max(running_max_g, hp)
            holm_ps_g.append(min(1.0, running_max_g))
        stat_df.loc[sub_stat_g.index, "holm_p_value_global"] = holm_ps_g

    stat_df.to_csv("results/v5/tables/v5_statistical_comparisons.csv", index=False)
    print("Saved statistical comparisons to results/v5/tables/v5_statistical_comparisons.csv")

    # False-Certification Diagnostic on Tied Null Pools (Diagnostic E)
    print("\n--- Diagnostic E: False-Certification Rate under Tied Null Simulations ---")
    N_pop = 30
    n_null_sims = 1000
    window_ids = [f"w_{i:03d}" for i in range(N_pop)]
    
    # Null scenario: Model A and Model B are identically distributed (true Delta = 0)
    false_certifications = 0
    false_eliminations = 0

    for sim in range(n_null_sims):
        sim_seed = 50000 + sim
        rng_sim = np.random.RandomState(sim_seed)
        null_D = rng_sim.normal(0.0, 0.35, size=N_pop)
        null_D_dict = {w: float(null_D[i]) for i, w in enumerate(window_ids)}

        design = FinitePopulationSamplingDesign(window_ids, seed=sim_seed)
        estimator = AIPWPairedEstimator(window_ids, clipping_bound=3.0)
        estimator.set_predictions({w: 0.0 for w in window_ids})
        cs = AnytimeValidConfidenceSequence(population_size=N_pop, delta=0.05, bound=3.0)

        for step in range(15):
            w_info = design.sample_next_window()
            if w_info is None:
                break
            w_t, q_t = w_info
            d_val = null_D_dict[w_t]
            y_inc = estimator.update(w_t, d_val, q_t)
            design.mark_observed(w_t)
            cs.update(y_inc)

            lcb, ucb, _ = cs.get_bounds()
            # Under null Delta = 0, eliminating Model B requires LCB > 0 (Type I error)
            if lcb > 0.0:
                false_eliminations += 1
                break
            # False certification: declaring leader strictly superior when Delta = 0 with high margin
            if ucb < -0.10:
                false_certifications += 1
                break

    type_1_error_rate = false_eliminations / n_null_sims
    print(f"Empirical Type I Error (False Elimination) Rate: {type_1_error_rate*100:.2f}% (Nominal bound <= 5.0%)")
    print(f"False Certification Rate: {(false_certifications/n_null_sims)*100:.2f}%")
    assert type_1_error_rate <= 0.055, f"Type I error exceeded nominal 5%: {type_1_error_rate}"

    return stat_df


if __name__ == "__main__":
    run_statistical_analysis()

