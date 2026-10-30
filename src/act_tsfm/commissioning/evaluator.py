"""Evaluates commissioning decisions (Global Selection & Regime Routing) on the Untouched Future Holdout."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any

MODELS = ["har", "garch", "chronos", "timesfm"]


class HoldoutCommissioningEvaluator:
    def __init__(self, holdout_reference_df: pd.DataFrame):
        self.holdout_df = holdout_reference_df.copy()
        
        # Precompute holdout oracle benchmarks
        # 1. Global oracle
        model_losses = self.holdout_df.groupby("model_id")["qlike_loss"].mean()
        self.oracle_global_model = str(model_losses.idxmin())
        self.oracle_global_loss = float(model_losses.min())
        
        # 2. Cell-specific routing oracle
        cell_model_losses = self.holdout_df.groupby(["cell_id", "model_id"])["qlike_loss"].mean()
        self.oracle_routing_map = {}
        self.oracle_routing_cell_losses = {}
        for c in self.holdout_df["cell_id"].unique():
            sub = cell_model_losses.loc[c]
            best_m = str(sub.idxmin())
            self.oracle_routing_map[c] = best_m
            self.oracle_routing_cell_losses[c] = float(sub.min())
            
        # Oracle routing total loss across holdout windows
        total_oracle_routing_loss = 0.0
        w_grouped = self.holdout_df.groupby("window_id")
        for w_id, w_rows in w_grouped:
            cell = w_rows["cell_id"].iloc[0]
            opt_m = self.oracle_routing_map[cell]
            opt_loss = float(w_rows[w_rows["model_id"] == opt_m]["qlike_loss"].iloc[0])
            total_oracle_routing_loss += opt_loss
        self.oracle_routing_loss = total_oracle_routing_loss / len(w_grouped)

    def evaluate(
        self, 
        global_rec: str, 
        routing_rec: dict[str, str],
        abstain: bool = False
    ) -> dict[str, Any]:
        """Evaluates chosen models on holdout windows."""
        # 1. Global regret
        global_loss = float(self.holdout_df[self.holdout_df["model_id"] == global_rec]["qlike_loss"].mean())
        global_regret = max(0.0, global_loss - self.oracle_global_loss)
        global_oracle_agreement = (global_rec == self.oracle_global_model)

        # 2. Routing regret
        w_grouped = self.holdout_df.groupby("window_id")
        total_routing_loss = 0.0
        cell_regrets = []
        routing_agreements = []
        
        for w_id, w_rows in w_grouped:
            cell = w_rows["cell_id"].iloc[0]
            chosen_m = routing_rec.get(cell, global_rec)
            m_loss_row = w_rows[w_rows["model_id"] == chosen_m]
            if len(m_loss_row):
                loss_val = float(m_loss_row["qlike_loss"].iloc[0])
            else:
                loss_val = float(w_rows["qlike_loss"].mean())
            total_routing_loss += loss_val
            
            opt_m = self.oracle_routing_map.get(cell, global_rec)
            routing_agreements.append(chosen_m == opt_m)

        routing_loss = total_routing_loss / len(w_grouped)
        routing_regret = max(0.0, routing_loss - self.oracle_routing_loss)
        routing_agreement = float(np.mean(routing_agreements))
        
        # Worst regime regret
        regime_losses = []
        for regime in ["calm", "normal", "stress"]:
            r_rows = self.holdout_df[self.holdout_df["regime"] == regime]
            if len(r_rows):
                r_loss = float(r_rows[r_rows["model_id"] == global_rec]["qlike_loss"].mean())
                r_opt = float(r_rows.groupby("model_id")["qlike_loss"].mean().min())
                regime_losses.append(r_loss - r_opt)
        worst_regime_regret = max(0.0, float(max(regime_losses))) if regime_losses else 0.0

        return {
            "global_recommendation": global_rec,
            "oracle_global_model": self.oracle_global_model,
            "global_loss": global_loss,
            "global_regret": global_regret,
            "global_oracle_agreement": global_oracle_agreement,
            "routing_loss": routing_loss,
            "routing_regret": routing_regret,
            "routing_oracle_agreement": routing_agreement,
            "worst_regime_regret": worst_regime_regret,
            "abstain": abstain
        }

