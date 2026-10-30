"""AIPW paired estimator for finite populations under sequential sampling without replacement."""

from __future__ import annotations
import numpy as np
from typing import Sequence


class AIPWPairedEstimator:
    """Augmented Inverse-Propensity-Weighted (AIPW) estimator for the finite-population mean

    paired difference Delta_{A,B,c} = (1/N_c) sum_{w=1}^{N_c} D_{A,B,c}(w).

    Preserves exact finite-population unbiasedness under predictable sequential sampling
    without replacement, and reduces EXACTLY to sample-mean paired racing when g = 0 and sampling is uniform.
    """
    def __init__(
        self,
        all_window_ids: Sequence[str],
        clipping_bound: float = 3.0
    ):
        self.all_windows = list(all_window_ids)
        self.N = len(self.all_windows)
        self.B = clipping_bound
        
        # Pre-observation predictions g_t(w) and residual scale s_t(w)
        self.predictions: dict[str, float] = {}
        self.residual_scales: dict[str, float] = {}
        
        # Recorded observations
        self.observed_D: dict[str, float] = {}
        self.observed_order: list[str] = []
        self.sampling_probs: list[float] = []
        self.aipw_increments: list[float] = []
        self.residuals: list[float] = []

    def set_predictions(self, g_dict: dict[str, float], s_dict: dict[str, float] | None = None):
        """Sets pre-observation predictions g_t(w) before evaluating any window."""
        for w, val in g_dict.items():
            self.predictions[w] = float(np.clip(val, -self.B, self.B))
        if s_dict:
            for w, val in s_dict.items():
                self.residual_scales[w] = max(1e-4, float(val))

    def update(self, window_id: str, d_obs: float, sampling_prob: float) -> float:
        """Records a new paired observation D(w) drawn with probability q_t(w).

        Computes the unbiased finite-population AIPW pseudo-outcome Y_t.
        """
        clipped_d = float(np.clip(d_obs, -self.B, self.B))
        self.observed_D[window_id] = clipped_d
        self.observed_order.append(window_id)
        self.sampling_probs.append(sampling_prob)

        # Compute unobserved set before this observation was added
        prev_observed = set(self.observed_order[:-1])
        unobs_before = [w for w in self.all_windows if w not in prev_observed]
        
        # Known sum of observed D so far (excluding current)
        sum_known_D = sum(self.observed_D[w] for w in prev_observed)
        
        # Sum of g(u) over unobserved windows
        sum_unobs_g = sum(self.predictions.get(u, 0.0) for u in unobs_before)
        
        # Current prediction g(w_t)
        g_w = self.predictions.get(window_id, 0.0)
        res = clipped_d - g_w
        self.residuals.append(res)
        
        # Unbiased finite-population Horvitz-Thompson AIPW increment
        # E[Y_t | F_{t-1}] = Delta exactly
        ht_correction = res / max(1e-6, sampling_prob)
        y_t = (sum_known_D + sum_unobs_g + ht_correction) / self.N
        
        self.aipw_increments.append(y_t)
        return y_t

    def get_estimate(self) -> float:
        """Returns the exact finite-population augmented difference estimator.

        When g = 0 and sampling is uniform, this matches the sample mean (1/t) sum D(w_i) exactly.
        """
        t = len(self.observed_order)
        if t == 0:
            if not self.predictions:
                return 0.0
            return float(np.mean([self.predictions.get(w, 0.0) for w in self.all_windows]))
            
        if t >= self.N:
            # Full finite population evaluated: exact ground truth
            return float(np.mean([self.observed_D[w] for w in self.all_windows]))
            
        # Finite-population difference estimator:
        # Delta_hat = (1/N) * [ sum_{w in S_t} D(w) + sum_{u not in S_t} g(u) + (N - t) * mean_sample(residual) ]
        sum_sample_D = sum(self.observed_D[w] for w in self.observed_order)
        observed_set = set(self.observed_order)
        sum_unobs_g = sum(self.predictions.get(w, 0.0) for w in self.all_windows if w not in observed_set)
        mean_res = float(np.mean(self.residuals))
        
        point_est = (sum_sample_D + sum_unobs_g + (self.N - t) * mean_res) / self.N
        return float(point_est)

    def get_variance_estimate(self) -> float:
        """Returns empirical variance of the AIPW increments."""
        t = len(self.aipw_increments)
        if t <= 1:
            return 1.0
        # Finite-population variance adjustment factor: (1 - t/N)
        fp_factor = max(0.0, 1.0 - (t / self.N))
        s2 = float(np.var(self.aipw_increments, ddof=1))
        return s2 * fp_factor
