"""Never-idle top-two Thompson sampling fallback policy."""

from __future__ import annotations
import numpy as np


class ThompsonFallbackPolicy:
    """Never-idle fallback: when lookahead KG* acquisition values are zero or negative,

    identifies provisional leader and strongest uncertified challenger via Thompson draws
    and evaluates them on the next window drawn according to the valid sampling design.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select_fallback_bundle(
        self,
        active_arms: list[str],
        means: dict[str, float],
        variances: dict[str, float],
        sampling_design,
        evaluator
    ) -> tuple[list[str], str] | None:
        """Selects leader + top challenger and the next window drawn from the sampling design."""
        if not active_arms:
            return None

        # Thompson sample draw for each active arm
        draws = {}
        for m in active_arms:
            mu = means.get(m, 0.35)
            s = np.sqrt(max(1e-4, variances.get(m, 0.20)))
            draws[m] = float(self.rng.normal(mu, s))

        sorted_arms = sorted(active_arms, key=lambda m: draws[m])
        leader = sorted_arms[0]
        challenger = sorted_arms[1] if len(sorted_arms) > 1 else leader

        # Sample next window from remaining unobserved windows in this cell
        next_w_info = sampling_design.sample_next_window()
        if next_w_info is None:
            return None
        w_id, _ = next_w_info

        # Form paired bundle if possible
        if leader != challenger:
            pair = [leader, challenger]
            unrevealed = [m for m in pair if (m, w_id) not in evaluator.revealed_observations]
            if unrevealed and evaluator.can_afford(unrevealed, w_id):
                return unrevealed, w_id

        # Fallback to single arm
        for m in [leader, challenger]:
            if (m, w_id) not in evaluator.revealed_observations and evaluator.can_afford([m], w_id):
                return [m], w_id

        return None

