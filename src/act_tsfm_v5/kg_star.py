"""Lookahead KG* bundle acquisition scorer."""

from __future__ import annotations
import numpy as np


class KGStarScorer:
    """Computes lookahead Knowledge Gradient (KG*) acquisition score:

        A_{KG*}(a) = max_{1 <= k <= K} VoI(a^x k) / (k * C(a))
    """
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth

    def score_bundle(
        self,
        models: list[str],
        window_id: str,
        cost: float,
        predicted_gap: float,
        residual_uncertainty: float
    ) -> float:
        """Scores candidate bundle a = (models, window_id)."""
        if cost <= 0:
            cost = 0.01

        # Expected reduction in decision regret between leader and challenger
        voi = residual_uncertainty * np.exp(-abs(predicted_gap) / (residual_uncertainty + 1e-4))
        
        # Lookahead k=1 and k=2
        best_rate = voi / cost
        if self.max_depth >= 2:
            voi_k2 = 1.414 * voi
            rate_k2 = voi_k2 / (2.0 * cost)
            best_rate = max(best_rate, rate_k2)

        return float(best_rate)

