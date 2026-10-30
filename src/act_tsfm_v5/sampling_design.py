"""Sampling design for finite-population sequential testing without replacement."""

from __future__ import annotations
import numpy as np
from typing import Sequence


class FinitePopulationSamplingDesign:
    """Manages sequential sampling of unique test windows from a finite cell population.

    Prevents duplicate evaluations and maintains predictable sampling probabilities.
    """
    def __init__(
        self, 
        window_ids: Sequence[str], 
        min_propensity: float = 0.05,
        lambda_init: float = 0.50,
        lambda_min: float = 0.10,
        seed: int = 42
    ):
        self.all_windows = list(window_ids)
        self.N = len(self.all_windows)
        self.w_to_idx = {w: i for i, w in enumerate(self.all_windows)}
        self.min_propensity = min_propensity
        self.lambda_t = lambda_init
        self.lambda_min = lambda_min
        self.rng = np.random.RandomState(seed)
        
        self.observed_windows: set[str] = set()
        self.observation_order: list[str] = []

    @property
    def n_observed(self) -> int:
        return len(self.observed_windows)

    @property
    def n_unobserved(self) -> int:
        return self.N - self.n_observed

    @property
    def unobserved_windows(self) -> list[str]:
        return [w for w in self.all_windows if w not in self.observed_windows]

    def compute_sampling_probabilities(
        self, 
        residuals_std: dict[str, float] | None = None,
        costs: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Computes predictable sampling distribution q_t(w) over remaining unobserved windows.

        Uses cost-adjusted Neyman allocation:
            pi*(w) proportional to s_t(w) / sqrt(C(w))
        mixed with uniform exploration:
            q_t(w) = (1 - lambda_t) * pi*(w) + lambda_t * (1 / N_unobserved)
        subject to minimum propensity floor.
        """
        unobs = self.unobserved_windows
        k = len(unobs)
        if k == 0:
            return {}
        if k == 1:
            return {unobs[0]: 1.0}

        if residuals_std is None or len(residuals_std) == 0:
            # Uniform distribution over unobserved
            p_uni = 1.0 / k
            return {w: p_uni for w in unobs}

        # Compute cost-adjusted scores
        scores = np.zeros(k, dtype=np.float64)
        for i, w in enumerate(unobs):
            s = max(1e-4, float(residuals_std.get(w, 0.20)))
            c = max(1e-4, float(costs.get(w, 0.05))) if costs else 1.0
            scores[i] = s / np.sqrt(c)

        sum_scores = np.sum(scores)
        if sum_scores <= 1e-9:
            neyman_probs = np.ones(k) / k
        else:
            neyman_probs = scores / sum_scores

        # Uniform mixture
        uni_prob = 1.0 / k
        mixed_probs = (1.0 - self.lambda_t) * neyman_probs + self.lambda_t * uni_prob

        # Enforce minimum propensity floor
        floor = self.min_propensity / k
        floored_probs = np.maximum(mixed_probs, floor)
        final_probs = floored_probs / np.sum(floored_probs)

        return {w: float(final_probs[i]) for i, w in enumerate(unobs)}

    def sample_next_window(
        self, 
        residuals_std: dict[str, float] | None = None,
        costs: dict[str, float] | None = None
    ) -> tuple[str, float] | None:
        """Draws the next window to evaluate without replacement.

        Returns (window_id, sampling_probability_q_t).
        """
        prob_dict = self.compute_sampling_probabilities(residuals_std, costs)
        if not prob_dict:
            return None

        windows = list(prob_dict.keys())
        probs = np.array([prob_dict[w] for w in windows], dtype=np.float64)
        probs = probs / np.sum(probs)

        chosen_idx = self.rng.choice(len(windows), p=probs)
        chosen_w = windows[chosen_idx]
        chosen_p = probs[chosen_idx]

        return chosen_w, chosen_p

    def mark_observed(self, window_id: str):
        """Marks a window as evaluated so it cannot be selected again."""
        if window_id not in self.observed_windows:
            self.observed_windows.add(window_id)
            self.observation_order.append(window_id)
            # Slowly decay lambda_t toward lambda_min as evidence accumulates
            self.lambda_t = max(self.lambda_min, self.lambda_t * 0.98)

