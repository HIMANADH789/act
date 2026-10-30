"""Reveal-only evaluation engine enforcing information-hiding boundaries, cost accounting and budget limits."""

from __future__ import annotations
import copy
from typing import Any
import pandas as pd
import numpy as np


class RevealOnlyEvaluator:
    """Evaluator that holds the full ground-truth reference matrix in-memory

    but only exposes requested atomic (m,w) or bundle (S,w) evaluations.
    Enforces cost budgets, duplicate prevention, and logs an immutable ledger.
    """
    def __init__(self, reference_df: pd.DataFrame, budget_cap: float):
        self._reference = reference_df.copy()
        # Create lookup index on (model_id, window_id)
        self._lookup = {
            (row["model_id"], row["window_id"]): row.to_dict()
            for _, row in self._reference.iterrows()
        }
        self.budget_cap = float(budget_cap)
        self.spent_cost = 0.0
        self.revealed_observations: dict[tuple[str, str], dict[str, Any]] = {}
        self.ledger: list[dict[str, Any]] = []

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.budget_cap - self.spent_cost)

    @property
    def is_budget_exhausted(self) -> bool:
        return self.spent_cost >= self.budget_cap

    def can_afford(self, model_ids: list[str], window_id: str) -> bool:
        needed_cost = 0.0
        for m in model_ids:
            if (m, window_id) in self.revealed_observations:
                continue
            entry = self._lookup.get((m, window_id))
            if entry is not None:
                needed_cost += entry["inference_cost"]
        return (self.spent_cost + needed_cost) <= (self.budget_cap + 1e-9)

    def measure_bundle_cost(self, model_ids: list[str], window_id: str) -> float:
        cost = 0.0
        for m in model_ids:
            if (m, window_id) in self.revealed_observations:
                continue
            entry = self._lookup.get((m, window_id))
            if entry is not None:
                cost += entry["inference_cost"]
        return cost

    def evaluate_bundle(
        self, 
        model_ids: list[str], 
        window_id: str, 
        step_idx: int = 0
    ) -> list[dict[str, Any]]:
        """Evaluates a bundle a = (S, w) where S is a non-empty subset of candidate models

        evaluated on the common window w.
        Returns the revealed observation records for newly evaluated (m,w) pairs.
        Rejects duplicates and enforces budget.
        """
        if not model_ids:
            return []

        # Calculate cost for unrevealed models in bundle
        unrevealed = [m for m in model_ids if (m, window_id) not in self.revealed_observations]
        if not unrevealed:
            # Already revealed all
            return [copy.deepcopy(self.revealed_observations[(m, window_id)]) for m in model_ids]

        cost_to_charge = sum(self._lookup[(m, window_id)]["inference_cost"] for m in unrevealed)
        if self.spent_cost + cost_to_charge > self.budget_cap + 1e-9:
            # Cannot exceed budget
            return []

        outcomes = []
        for m in unrevealed:
            record = copy.deepcopy(self._lookup[(m, window_id)])
            self.spent_cost += record["inference_cost"]
            self.revealed_observations[(m, window_id)] = record
            
            ledger_entry = {
                "step": step_idx,
                "model_id": m,
                "window_id": window_id,
                "asset_id": record["asset_id"],
                "regime": record["regime"],
                "cell_id": record["cell_id"],
                "horizon": record["horizon"],
                "forecast_origin": record["forecast_origin"],
                "qlike_loss": record["qlike_loss"],
                "cost_charged": record["inference_cost"],
                "cumulative_cost": self.spent_cost,
                "remaining_budget": self.remaining_budget
            }
            self.ledger.append(ledger_entry)
            outcomes.append(record)

        # Return all requested models (both newly revealed and previously revealed)
        return [copy.deepcopy(self.revealed_observations[(m, window_id)]) for m in model_ids]

    def evaluate_atomic(self, model_id: str, window_id: str, step_idx: int = 0) -> dict[str, Any] | None:
        """Evaluates an atomic pair (m,w)."""
        res = self.evaluate_bundle([model_id], window_id, step_idx=step_idx)
        return res[0] if res else None

    def get_ledger_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.ledger)

