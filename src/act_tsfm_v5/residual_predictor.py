"""Residual scale predictor for cost-aware Neyman allocation."""

from __future__ import annotations
import numpy as np


class ResidualScalePredictor:
    """Predicts residual scale s_t(w) approx SD[D(w) - g_t(w)] for window w

    using past observed residuals strictly prequentially.
    """
    def __init__(self, default_scale: float = 0.25):
        self.default_scale = default_scale
        self.observed_abs_residuals: list[float] = []
        self.cell_abs_residuals: dict[str, list[float]] = {}

    def update(self, cell_id: str, actual_residual: float):
        """Prequentially updates residual history."""
        abs_res = float(abs(actual_residual))
        self.observed_abs_residuals.append(abs_res)
        if cell_id not in self.cell_abs_residuals:
            self.cell_abs_residuals[cell_id] = []
        self.cell_abs_residuals[cell_id].append(abs_res)

    def predict_scale(self, cell_id: str, window_id: str, base_var: float = 0.20) -> float:
        """Predicts residual standard error s_t(w) for an unobserved window."""
        cell_history = self.cell_abs_residuals.get(cell_id, [])
        if len(cell_history) >= 3:
            return float(np.mean(cell_history[-10:]) + 0.05)
        elif len(self.observed_abs_residuals) >= 3:
            return float(np.mean(self.observed_abs_residuals[-15:]) + 0.05)
        return float(np.sqrt(base_var) + 0.05)

