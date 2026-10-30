"""V4 In-Context Transformer adapter for predicting paired performance differences."""

from __future__ import annotations
import os
import torch
import numpy as np
import pandas as pd
from typing import Sequence

from act_tsfm.v4.axial_transformer import InContextAxialTransformer
from act_tsfm.model_descriptors import (
    extract_model_descriptor_matrix,
    extract_cell_descriptor_matrix,
    CELL_LIST
)

PRIMARY_MODELS = ["har", "garch", "chronos", "timesfm"]


class V4ContextAdapter:
    """Adapts the pretrained V4 in-context transformer to predict paired loss differences

    g_t(w) approx D_{A,B,c}(w) = L_A(w) - L_B(w) before expensive evaluations.
    """
    def __init__(
        self,
        checkpoint_path: str = "artifacts/checkpoints/v4_pretrained.pt",
        device: str = "cuda",
        model_list: Sequence[str] | None = None
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.models = list(model_list) if model_list is not None else PRIMARY_MODELS
        self.model_desc = extract_model_descriptor_matrix()
        self.cell_desc: dict[str, np.ndarray] | None = None
        
        # Initialize architecture
        self.net = InContextAxialTransformer(d_model=64, n_layers=3, n_heads=4).to(self.device)
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
        self.net.eval()

    def init_descriptors(self, available_windows: pd.DataFrame):
        if self.cell_desc is None:
            self.cell_desc = extract_cell_descriptor_matrix(available_windows)

    def predict_paired_differences(
        self,
        ledger: list[dict],
        auxiliary_losses: pd.DataFrame | None = None
    ) -> dict[tuple[str, str, str], float]:
        """Generates predictions g_t(m1, m2, cell) for all pairs and cells.

        Returns {(m1, m2, cell): predicted_difference}.
        """
        M = len(PRIMARY_MODELS)
        C = len(CELL_LIST)
        
        m_t = torch.tensor(np.array([self.model_desc[m] for m in PRIMARY_MODELS]), dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.tensor(np.array([self.cell_desc[c] for c in CELL_LIST]), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        obs_arr = np.zeros((1, M, C, 3), dtype=np.float32)
        counts = np.zeros((M, C), dtype=int)
        
        # 1. Incorporate auxiliary cheap model evidence if provided
        if auxiliary_losses is not None and len(auxiliary_losses):
            for _, row in auxiliary_losses.iterrows():
                m = row["model_id"]
                if m in PRIMARY_MODELS:
                    m_idx = PRIMARY_MODELS.index(m)
                    c_idx = CELL_LIST.index(row["cell_id"])
                    loss = float(row["qlike_loss"])
                    obs_arr[0, m_idx, c_idx, 0] = (obs_arr[0, m_idx, c_idx, 0] * counts[m_idx, c_idx] + loss) / (counts[m_idx, c_idx] + 1)
                    obs_arr[0, m_idx, c_idx, 1] = float(row.get("inference_cost", 0.02))
                    obs_arr[0, m_idx, c_idx, 2] = 1.0
                    counts[m_idx, c_idx] += 1
                    
        # 2. Incorporate ledger observations
        for entry in ledger:
            m = entry["model_id"]
            if m in PRIMARY_MODELS:
                m_idx = PRIMARY_MODELS.index(m)
                c_idx = CELL_LIST.index(entry["cell_id"])
                loss = float(entry["qlike_loss"])
                obs_arr[0, m_idx, c_idx, 0] = (obs_arr[0, m_idx, c_idx, 0] * counts[m_idx, c_idx] + loss) / (counts[m_idx, c_idx] + 1)
                obs_arr[0, m_idx, c_idx, 1] = float(entry["cost_charged"])
                obs_arr[0, m_idx, c_idx, 2] = 1.0
                counts[m_idx, c_idx] += 1

        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
        w_tok = torch.zeros((1, M, C), dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_mean, pred_logvar, _ = self.net(m_t, c_t, obs_t, w_tok)
            p_mean = pred_mean[0].cpu().numpy()
            
        # Compute paired differences
        paired_dict = {}
        for m1 in self.models:
            for m2 in self.models:
                if m1 in PRIMARY_MODELS and m2 in PRIMARY_MODELS:
                    i1 = PRIMARY_MODELS.index(m1)
                    i2 = PRIMARY_MODELS.index(m2)
                    for c_idx, c in enumerate(CELL_LIST):
                        diff = float(p_mean[i1, c_idx] - p_mean[i2, c_idx])
                        paired_dict[(m1, m2, c)] = diff
                else:
                    # Generic heuristic for expanded models not in primary 4
                    for c in CELL_LIST:
                        paired_dict[(m1, m2, c)] = 0.0
                        
        return paired_dict

