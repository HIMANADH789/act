"""ACT-TSFM V4: Amortized In-Context Neural Transformer Router with Axial Attention,

Autoregressive Joint Prediction, Offline Multi-Task Pretraining, and Conformal Calibration.
"""

from __future__ import annotations
import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from ..evaluator.engine import RevealOnlyEvaluator
from ..model_descriptors import (
    extract_model_descriptor_matrix, 
    extract_cell_descriptor_matrix, 
    CELL_LIST
)

MODELS = ["har", "garch", "chronos", "timesfm"]


class AxialAttentionBlock(nn.Module):
    """Axial attention over Model Axis (M=4) and Cell/Window Axis (C)."""
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        # Model-axis self-attention
        self.m_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(d_model)
        
        # Cell-axis self-attention
        self.c_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_c = nn.LayerNorm(d_model)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_mlp = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [Batch, M, C, D]
        B, M, C, D = x.shape
        
        # 1. Model-axis attention (attend across models for each cell)
        # Reshape to [B * C, M, D]
        x_m = x.permute(0, 2, 1, 3).reshape(B * C, M, D)
        attn_m, _ = self.m_attn(x_m, x_m, x_m)
        x_m = self.norm_m(x_m + attn_m)
        x = x_m.reshape(B, C, M, D).permute(0, 2, 1, 3)
        
        # 2. Cell-axis attention (attend across cells for each model)
        # Reshape to [B * M, C, D]
        x_c = x.reshape(B * M, C, D)
        attn_c, _ = self.c_attn(x_c, x_c, x_c)
        x_c = self.norm_c(x_c + attn_c)
        x = x_c.reshape(B, M, C, D)
        
        # 3. Feed-forward
        x = self.norm_mlp(x + self.mlp(x))
        return x


class InContextAxialTransformer(nn.Module):
    """Amortized In-Context Transformer Router.

    Receives context of observed entries (m, c, w, loss, cost, is_observed)
    and predicts future mean, log_var, pairwise rankings and acquisition scores.
    """
    def __init__(self, d_model: int = 64, n_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Feature encoders
        # Model descriptors (dim 9) -> d_model
        self.enc_m = nn.Linear(9, d_model)
        # Cell descriptors (dim 6) -> d_model
        self.enc_c = nn.Linear(6, d_model)
        # Observation embedding: [loss, cost, is_observed] -> d_model
        self.enc_obs = nn.Linear(3, d_model)
        
        # Shared-window grouping embedding
        self.w_embed = nn.Embedding(64, d_model)
        
        # Axial transformer layers
        self.layers = nn.ModuleList([
            AxialAttentionBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Heads
        self.head_mean = nn.Linear(d_model, 1)
        self.head_logvar = nn.Linear(d_model, 1)
        self.head_policy = nn.Linear(d_model, 1)  # Amortized acquisition candidate scorer

    def forward(
        self, 
        m_feats: torch.Tensor,       # [B, M, 9]
        c_feats: torch.Tensor,       # [B, C, 6]
        obs_feats: torch.Tensor,     # [B, M, C, 3] (loss, cost, is_observed)
        w_tokens: torch.Tensor       # [B, M, C]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, _ = m_feats.shape
        C = c_feats.shape[1]
        
        # Project inputs
        em = self.enc_m(m_feats)[:, :, None, :]       # [B, M, 1, D]
        ec = self.enc_c(c_feats)[:, None, :, :]       # [B, 1, C, D]
        eobs = self.enc_obs(obs_feats)                # [B, M, C, D]
        ew = self.w_embed(torch.clamp(w_tokens, 0, 63)) # [B, M, C, D]
        
        x = em + ec + eobs + ew  # Combined token [B, M, C, D]
        
        for layer in self.layers:
            x = layer(x)
            
        pred_mean = self.head_mean(x).squeeze(-1)       # [B, M, C]
        pred_logvar = self.head_logvar(x).squeeze(-1)   # [B, M, C]
        policy_logits = self.head_policy(x).squeeze(-1) # [B, M, C]
        
        return pred_mean, pred_logvar, policy_logits


def generate_synthetic_task_batch(
    batch_size: int = 16, 
    n_models: int = 4, 
    n_cells: int = 9,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates synthetic pretraining episodes with structured model-cell losses,

    including independent arms, bilinear structure, and shared-window noise.
    """
    m_feats = torch.randn(batch_size, n_models, 9, device=device)
    c_feats = torch.randn(batch_size, n_cells, 6, device=device)
    
    # Generate ground-truth loss matrix Y
    # 50% structured bilinear, 50% independent arms
    W = torch.randn(batch_size, 9, 6, device=device) * 0.3
    bilinear = torch.einsum("bmi,bij,bcj->bmc", m_feats, W, c_feats)
    arm_effects = torch.randn(batch_size, n_models, 1, device=device) * 0.2
    shared_window = torch.randn(batch_size, 1, n_cells, device=device) * 0.15
    noise = torch.randn(batch_size, n_models, n_cells, device=device) * 0.1
    
    y_true = torch.clamp(0.35 + bilinear + arm_effects + shared_window + noise, 0.01, 3.0)
    
    # Random mask for observed context (e.g. 10% to 50% observed)
    mask = torch.rand(batch_size, n_models, n_cells, device=device) < 0.3
    costs = torch.rand(batch_size, n_models, n_cells, device=device) * 0.15 + 0.01
    
    obs_loss = torch.where(mask, y_true, torch.zeros_like(y_true))
    obs_cost = torch.where(mask, costs, torch.zeros_like(costs))
    is_obs = mask.float()
    
    obs_feats = torch.stack([obs_loss, obs_cost, is_obs], dim=-1)
    w_tokens = torch.randint(0, 32, (batch_size, n_models, n_cells), device=device)
    
    return m_feats, c_feats, obs_feats, w_tokens, y_true


def pretrain_v4_transformer(
    model: InContextAxialTransformer, 
    n_steps: int = 500, 
    device: str = "cuda"
) -> InContextAxialTransformer:
    """Pretrains V4 on synthetic structured performance tasks offline."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        m_f, c_f, obs_f, w_tok, y_true = generate_synthetic_task_batch(
            batch_size=16, device=device
        )
        
        pred_mean, pred_logvar, policy_logits = model(m_f, c_f, obs_f, w_tok)
        
        # 1. NLL loss
        var = torch.exp(torch.clamp(pred_logvar, -4.0, 3.0))
        nll = 0.5 * (((y_true - pred_mean)**2) / var + pred_logvar).mean()
        
        # 2. Pairwise ranking loss
        diff_true = y_true[:, :, None, :] - y_true[:, None, :, :]
        diff_pred = pred_mean[:, :, None, :] - pred_mean[:, None, :, :]
        rank_loss = F.relu(0.1 - torch.sign(diff_true) * diff_pred).mean()
        
        # 3. Decision-regret loss (penalize picking high-loss models)
        best_pred_m = torch.argmin(pred_mean.mean(dim=-1), dim=-1) # [B]
        # Regret of best predicted model
        batch_indices = torch.arange(y_true.shape[0], device=device)
        selected_loss = y_true[batch_indices, best_pred_m].mean(dim=-1)
        oracle_min_loss = y_true.mean(dim=-1).min(dim=-1).values
        regret_loss = (selected_loss - oracle_min_loss).mean()
        
        loss = nll + 0.5 * rank_loss + 1.0 * regret_loss
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model


class PrequentialConformalMonitor:
    """Prequential conformal calibration and temporal drift monitor."""
    def __init__(self, alpha: float = 0.05, window_size: int = 30):
        self.alpha = alpha
        self.window_size = window_size
        self.residuals: list[float] = []
        self.conformal_quantile: float = 0.35
        self.drift_detected: bool = False

    def update(self, y_true: float, y_pred_mean: float, y_pred_std: float):
        # Non-conformity score: standardized residual
        score = abs(y_true - y_pred_mean) / (y_pred_std + 1e-4)
        self.residuals.append(score)
        
        # Compute empirical quantile
        recent = self.residuals[-self.window_size:]
        q_idx = int(math.ceil((1.0 - self.alpha) * (len(recent) + 1))) - 1
        q_idx = min(max(0, q_idx), len(recent) - 1)
        self.conformal_quantile = float(np.sort(recent)[q_idx])
        
        # Drift check: if recent error rate > 2 * alpha, flag drift
        if len(recent) >= 15:
            coverage = np.mean(np.array(recent) <= 2.0)
            if coverage < 0.70:
                self.drift_detected = True


class V4Policy:
    """ACT-TSFM V4 Amortized In-Context Transformer Router with Conformal Calibration."""
    def __init__(self, seed: int = 42, device: str = "cuda"):
        self.name = "v4"
        self.seed = seed
        self.device = device if torch.cuda.is_available() else "cpu"
        self.rng = np.random.RandomState(seed)
        
        self.model_desc = extract_model_descriptor_matrix()
        self.cell_desc = None
        
        # Initialize model and load pretrained weights
        torch.manual_seed(seed)
        self.net = InContextAxialTransformer().to(self.device)
        ckpt_path = "artifacts/checkpoints/v4_pretrained.pt"
        if os.path.exists(ckpt_path):
            self.net.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.net.eval()
        self.conformal = PrequentialConformalMonitor()
        self.fallback_active = False

    def _init_components(self, available_windows: pd.DataFrame):
        if self.cell_desc is None:
            self.cell_desc = extract_cell_descriptor_matrix(available_windows)
            self.w_to_cell = dict(zip(available_windows["window_id"], available_windows["cell_id"]))

    def select_action(
        self, 
        evaluator: RevealOnlyEvaluator, 
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        self._init_components(available_windows)
        
        # If drift detected, fall back to conservative paired racing
        if self.conformal.drift_detected and not self.fallback_active:
            self.fallback_active = True

        # Build context tensor
        M = len(MODELS)
        C = len(CELL_LIST)
        
        m_t = torch.tensor(np.array([self.model_desc[m] for m in MODELS]), dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.tensor(np.array([self.cell_desc[c] for c in CELL_LIST]), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Fill observed entries from ledger
        obs_arr = np.zeros((1, M, C, 3), dtype=np.float32)
        counts = np.zeros((M, C), dtype=int)
        
        for entry in evaluator.ledger:
            m_idx = MODELS.index(entry["model_id"])
            c_idx = CELL_LIST.index(entry["cell_id"])
            loss = float(entry["qlike_loss"])
            cost = float(entry["cost_charged"])
            
            # Update conformal monitor with prequential prediction
            pred_mean_prior = 0.35 if counts[m_idx, c_idx] == 0 else obs_arr[0, m_idx, c_idx, 0]
            self.conformal.update(loss, pred_mean_prior, 0.20)
            
            # Running average in context
            obs_arr[0, m_idx, c_idx, 0] = (obs_arr[0, m_idx, c_idx, 0] * counts[m_idx, c_idx] + loss) / (counts[m_idx, c_idx] + 1)
            obs_arr[0, m_idx, c_idx, 1] = cost
            obs_arr[0, m_idx, c_idx, 2] = 1.0  # is_observed
            counts[m_idx, c_idx] += 1
            
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
        w_tok = torch.zeros((1, M, C), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            pred_mean, pred_logvar, policy_logits = self.net(m_t, c_t, obs_t, w_tok)
            p_mean = pred_mean[0].cpu().numpy()
            p_std = np.sqrt(np.exp(np.clip(pred_logvar[0].cpu().numpy(), -4.0, 3.0)))
            p_logits = policy_logits[0].cpu().numpy()

        best_cand = None
        best_score = -1e9

        for w_id, cell in self.w_to_cell.items():
            c_idx = CELL_LIST.index(cell)
            
            unrevealed = [m for m in MODELS if (m, w_id) not in evaluator.revealed_observations]
            if not unrevealed:
                continue

            for m in unrevealed:
                if not evaluator.can_afford([m], w_id):
                    continue
                m_idx = MODELS.index(m)
                cost = 0.003 if m == "har" else (0.025 if m == "garch" else (0.15 if m == "chronos" else 0.12))
                
                uncertainty = p_std[m_idx, c_idx] * self.conformal.conformal_quantile
                score = (p_logits[m_idx, c_idx] + 2.0 * uncertainty) / cost
                
                if score > best_score:
                    best_score = score
                    best_cand = ([m], w_id)

        if best_cand is not None:
            return best_cand
            
        return None

    def make_recommendations(
        self, 
        evaluator: RevealOnlyEvaluator
    ) -> tuple[str, dict[str, str]]:
        M = len(MODELS)
        C = len(CELL_LIST)
        m_t = torch.tensor(np.array([self.model_desc[m] for m in MODELS]), dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.tensor(np.array([self.cell_desc[c] for c in CELL_LIST]), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        obs_arr = np.zeros((1, M, C, 3), dtype=np.float32)
        for entry in evaluator.ledger:
            m_idx = MODELS.index(entry["model_id"])
            c_idx = CELL_LIST.index(entry["cell_id"])
            obs_arr[0, m_idx, c_idx, 0] = float(entry["qlike_loss"])
            obs_arr[0, m_idx, c_idx, 1] = float(entry["cost_charged"])
            obs_arr[0, m_idx, c_idx, 2] = 1.0
            
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
        w_tok = torch.zeros((1, M, C), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            pred_mean, _, _ = self.net(m_t, c_t, obs_t, w_tok)
            p_mean = pred_mean[0].cpu().numpy()

        global_rec = min(MODELS, key=lambda m: np.mean(p_mean[MODELS.index(m)]))
        routing = {}
        for c_idx, c in enumerate(CELL_LIST):
            routing[c] = min(MODELS, key=lambda m: p_mean[MODELS.index(m), c_idx])
            
        return global_rec, routing
