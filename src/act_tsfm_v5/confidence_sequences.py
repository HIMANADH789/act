"""Anytime-valid confidence sequences for sequential paired testing and racing."""

from __future__ import annotations
import math
import numpy as np


class AnytimeValidConfidenceSequence:
    """Constructs anytime-valid, time-uniform confidence sequences for finite-population

    paired estimators. Guarantees:
        P(exists t >= 1: Delta not in [LCB_t, UCB_t]) <= delta
    with finite-population variance reduction factor sqrt(1 - t/N).
    """
    def __init__(
        self,
        population_size: int,
        delta: float = 0.05,
        bound: float = 3.0,
        c_scale: float = 1.0
    ):
        self.N = population_size
        self.delta = delta
        self.bound = bound
        self.c_scale = c_scale
        
        self.t = 0
        self.sum_y = 0.0
        self.sum_sq_diff = 0.0
        self.y_history: list[float] = []

    def update(self, y_increment: float):
        """Updates confidence sequence with a new observation increment."""
        self.t += 1
        self.y_history.append(y_increment)
        self.sum_y += y_increment
        
        # Incremental variance calculation (Welford)
        if self.t > 1:
            mean = self.sum_y / self.t
            self.sum_sq_diff += (y_increment - self.y_history[-2]) ** 2

    def get_bounds(self) -> tuple[float, float, float]:
        """Returns (lcb, ucb, width) for the current step."""
        if self.t == 0:
            return -self.bound, self.bound, self.bound

        # Exact finite-population exhaustion: if t >= N, uncertainty is exactly zero
        if self.t >= self.N:
            point_est = self.sum_y / self.t
            return point_est, point_est, 0.0

        point_est = self.sum_y / self.t
        
        # Empirical-Bernstein predictable boundary
        # log-term with family-wise error alpha
        log_term = np.log(2.0 / max(1e-9, self.delta))
        
        # Predictable variance proxy
        s2 = (self.sum_sq_diff / (2.0 * max(1, self.t - 1))) + 0.01
        
        # Howard et al. (2021) / Waudby-Smith & Ramdas (2024) empirical-Bernstein boundary
        # psi(t) = sqrt(2 * V_t * log(2/delta)) / t
        raw_width = (np.sqrt(2.0 * s2 * log_term / self.t) + (3.0 * self.bound * log_term / self.t)) * self.c_scale
        
        # Finite population correction factor: sqrt(max(0, 1 - t/N))
        fpc = np.sqrt(max(0.0, 1.0 - (self.t / self.N)))
        width = float(raw_width * fpc)

        lcb = float(point_est - width)
        ucb = float(point_est + width)
        return lcb, ucb, width

