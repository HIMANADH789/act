"""Conventional selection strategies: Fixed, Random, Risk-first, Uncertainty, UCB, Racing."""

from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from typing import Any
from ..evaluator.engine import RevealOnlyEvaluator


MODELS = ["har", "garch", "chronos", "timesfm"]


class BasePolicy:
    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        """Returns (selected_models, window_id) to evaluate next, or None if stopping."""
        raise NotImplementedError

    def make_recommendations(
        self, 
        evaluator: RevealOnlyEvaluator
    ) -> tuple[str, dict[str, str]]:
        """Returns (global_model, routing_dict {cell_id: model})."""
        ledger = evaluator.ledger
        if not ledger:
            # Fallback to har or garch
            return "garch", {c: "garch" for c in [
                "calm_h1", "calm_h5", "calm_h22",
                "normal_h1", "normal_h5", "normal_h22",
                "stress_h1", "stress_h5", "stress_h22"
            ]}
        
        df = pd.DataFrame(ledger)
        # Global recommendation: model with minimum overall mean loss
        mean_losses = df.groupby("model_id")["qlike_loss"].mean()
        for m in MODELS:
            if m not in mean_losses:
                mean_losses[m] = 1.0  # prior penalty if not evaluated
        global_rec = mean_losses.idxmin()

        # Routing recommendation per cell
        routing = {}
        cell_means = df.groupby(["cell_id", "model_id"])["qlike_loss"].mean()
        all_cells = [
            "calm_h1", "calm_h5", "calm_h22",
            "normal_h1", "normal_h5", "normal_h22",
            "stress_h1", "stress_h5", "stress_h22"
        ]
        for c in all_cells:
            if c in cell_means.index.levels[0]:
                sub = cell_means.loc[c]
                routing[c] = sub.idxmin()
            else:
                routing[c] = global_rec
        return str(global_rec), routing


class FixedBalancedPolicy(BasePolicy):
    """Evenly and round-robin cycles through all models and cells in fixed deterministic order."""
    def __init__(self, seed: int = 42):
        super().__init__("fixed", seed=seed)
        self.step = 0

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        all_w_ids = available_windows["window_id"].unique()
        n_windows = len(all_w_ids)
        
        # Cycle through windows and models
        for offset in range(n_windows * len(MODELS)):
            idx = (self.step + offset)
            m = MODELS[idx % len(MODELS)]
            w_id = all_w_ids[(idx // len(MODELS)) % n_windows]
            
            if (m, w_id) not in evaluator.revealed_observations:
                if evaluator.can_afford([m], w_id):
                    self.step = idx + 1
                    return [m], w_id
        return None


class RandomSelectionPolicy(BasePolicy):
    """Uniformly samples unobserved (m,w) pairs at random."""
    def __init__(self, seed: int = 42):
        super().__init__("random", seed=seed)

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        all_w_ids = available_windows["window_id"].values
        shuffled_w = self.rng.permutation(all_w_ids)
        shuffled_m = self.rng.permutation(MODELS)
        
        for w_id in shuffled_w:
            for m in shuffled_m:
                if (m, w_id) not in evaluator.revealed_observations:
                    if evaluator.can_afford([m], w_id):
                        return [m], w_id
        return None


class RiskFirstPolicy(BasePolicy):
    """Prioritizes stress regimes and high-volatility horizon windows to test downside risk."""
    def __init__(self, seed: int = 42):
        super().__init__("risk", seed=seed)

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        # Order windows: stress first, then normal, then calm; highest horizon first
        regime_prio = {"stress": 0, "normal": 1, "calm": 2}
        sorted_windows = available_windows.copy()
        sorted_windows["prio"] = sorted_windows["regime"].map(regime_prio)
        sorted_windows = sorted_windows.sort_values(["prio", "horizon"], ascending=[True, False])
        
        for _, row in sorted_windows.iterrows():
            w_id = row["window_id"]
            # Prefer foundation models and GARCH under stress
            for m in ["timesfm", "chronos", "garch", "har"]:
                if (m, w_id) not in evaluator.revealed_observations:
                    if evaluator.can_afford([m], w_id):
                        return [m], w_id
        return None


class UncertaintyOnlyPolicy(BasePolicy):
    """Allocates observations to models/cells with the highest empirical loss dispersion."""
    def __init__(self, seed: int = 42):
        super().__init__("uncertainty", seed=seed)

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        ledger = evaluator.ledger
        counts = {(m, c): 0 for m in MODELS for c in available_windows["cell_id"].unique()}
        for entry in ledger:
            key = (entry["model_id"], entry["cell_id"])
            counts[key] = counts.get(key, 0) + 1

        # Sort cells and models by lowest observation count (highest epistemic uncertainty)
        sorted_keys = sorted(counts.keys(), key=lambda k: counts[k])
        for m, cell in sorted_keys:
            cand_windows = available_windows[available_windows["cell_id"] == cell]
            for _, row in cand_windows.iterrows():
                w_id = row["window_id"]
                if (m, w_id) not in evaluator.revealed_observations:
                    if evaluator.can_afford([m], w_id):
                        return [m], w_id
        return None


class CostAwareUCBPolicy(BasePolicy):
    """Upper Confidence Bound (optimistic for loss reduction) divided by action cost."""
    def __init__(self, seed: int = 42, beta: float = 1.0):
        super().__init__("ucb", seed=seed)
        self.beta = beta

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        ledger = evaluator.ledger
        t = len(ledger) + 1
        
        # Calculate mean loss and count per model
        means = {m: 0.5 for m in MODELS}
        counts = {m: 0 for m in MODELS}
        for entry in ledger:
            m = entry["model_id"]
            counts[m] += 1
            
        df_l = pd.DataFrame(ledger) if ledger else pd.DataFrame()
        if len(df_l):
            m_means = df_l.groupby("model_id")["qlike_loss"].mean()
            for m in m_means.index:
                means[m] = float(m_means[m])

        # UCB score for loss minimization: Score = -mean + beta * sqrt(2*log(t)/(N+1)) / cost
        scores = {}
        for m in MODELS:
            n = counts[m]
            bonus = self.beta * np.sqrt(2.0 * np.log(t + 1) / (n + 1))
            # Optimistic estimate of negative loss
            cost = 0.02 if m in ["har", "garch"] else 0.15
            scores[m] = (-means[m] + bonus) / cost

        sorted_models = sorted(MODELS, key=lambda m: scores[m], reverse=True)
        for m in sorted_models:
            for w_id in available_windows["window_id"].values:
                if (m, w_id) not in evaluator.revealed_observations:
                    if evaluator.can_afford([m], w_id):
                        return [m], w_id
        return None


class RacingPolicy(BasePolicy):
    """Successive halving / elimination racing: evaluates candidate models in rounds,

    dropping statistically significantly inferior candidates.
    """
    def __init__(self, seed: int = 42):
        super().__init__("racing", seed=seed)
        self.active_models = list(MODELS)
        self.round_idx = 0

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        ledger = evaluator.ledger
        df = pd.DataFrame(ledger) if ledger else pd.DataFrame()
        
        # Check for elimination every 10 observations if multiple models remain
        if len(df) >= 12 and len(self.active_models) > 1:
            means = df.groupby("model_id")["qlike_loss"].mean()
            counts = df.groupby("model_id")["qlike_loss"].count()
            best_m = means.idxmin()
            best_mean = means[best_m]
            
            survivors = []
            for m in self.active_models:
                if m == best_m:
                    survivors.append(m)
                    continue
                m_loss = df[df["model_id"] == m]["qlike_loss"].values
                best_loss = df[df["model_id"] == best_m]["qlike_loss"].values
                n_pairs = min(len(m_loss), len(best_loss))
                if n_pairs >= 8:
                    diff = m_loss[:n_pairs] - best_loss[:n_pairs]
                    se = np.std(diff) / np.sqrt(n_pairs) + 1e-6
                    # Eliminate if diff > 2.0 * se
                    if np.mean(diff) - 2.0 * se > 0.05:
                        continue
                survivors.append(m)
            self.active_models = survivors if survivors else [best_m]

        # Select next window and evaluate active models
        all_w_ids = available_windows["window_id"].unique()
        for w_id in all_w_ids:
            # Try evaluating paired active models on this window
            unrevealed_active = [m for m in self.active_models if (m, w_id) not in evaluator.revealed_observations]
            if unrevealed_active and evaluator.can_afford(unrevealed_active, w_id):
                return unrevealed_active, w_id
            # Or single active model
            for m in self.active_models:
                if (m, w_id) not in evaluator.revealed_observations and evaluator.can_afford([m], w_id):
                    return [m], w_id
        return None


def get_baseline_policy(name: str, seed: int = 42) -> BasePolicy:
    if name == "fixed":
        return FixedBalancedPolicy(seed=seed)
    elif name == "random":
        return RandomSelectionPolicy(seed=seed)
    elif name == "risk":
        return RiskFirstPolicy(seed=seed)
    elif name == "uncertainty":
        return UncertaintyOnlyPolicy(seed=seed)
    elif name == "ucb":
        return CostAwareUCBPolicy(seed=seed)
    elif name == "racing":
        return RacingPolicy(seed=seed)
    else:
        raise ValueError(f"Unknown baseline: {name}")

