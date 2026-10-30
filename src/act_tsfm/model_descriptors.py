"""Model and cell descriptor representations for performance mapping."""

from __future__ import annotations
import numpy as np
import pandas as pd

MODEL_STATIC_FEATURES = {
    "har": {
        "family_stat": 1.0,
        "family_ml": 0.0,
        "family_foundation": 0.0,
        "log_param_count": np.log10(4.0),
        "context_length": 22.0,
        "is_pretrained": 0.0,
        "base_cost": 0.003
    },
    "garch": {
        "family_stat": 1.0,
        "family_ml": 0.0,
        "family_foundation": 0.0,
        "log_param_count": np.log10(3.0),
        "context_length": 250.0,
        "is_pretrained": 0.0,
        "base_cost": 0.025
    },
    "chronos": {
        "family_stat": 0.0,
        "family_ml": 0.0,
        "family_foundation": 1.0,
        "log_param_count": np.log10(8_000_000.0),
        "context_length": 256.0,
        "is_pretrained": 1.0,
        "base_cost": 0.150
    },
    "timesfm": {
        "family_stat": 0.0,
        "family_ml": 0.0,
        "family_foundation": 1.0,
        "log_param_count": np.log10(200_000_000.0),
        "context_length": 256.0,
        "is_pretrained": 1.0,
        "base_cost": 0.120
    }
}

MODELS = ["har", "garch", "chronos", "timesfm"]

CELL_LIST = [
    "calm_h1", "calm_h5", "calm_h22",
    "normal_h1", "normal_h5", "normal_h22",
    "stress_h1", "stress_h5", "stress_h22"
]


def extract_cell_descriptor_matrix(windows_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Computes median continuous descriptors for each of the 9 cells."""
    cell_descriptors = {}
    for cell in CELL_LIST:
        sub = windows_df[windows_df["cell_id"] == cell]
        if len(sub) == 0:
            regime, h_str = cell.split("_h")
            h = float(h_str)
            regime_val = 0.0 if regime == "calm" else (0.5 if regime == "normal" else 1.0)
            cell_descriptors[cell] = np.array([regime_val, np.log(h), 0.01, 0.0, -0.05, 0.0], dtype=np.float32)
        else:
            regime = sub["regime"].iloc[0]
            regime_val = 0.0 if regime == "calm" else (0.5 if regime == "normal" else 1.0)
            h = float(sub["horizon"].iloc[0])
            trail_vol = float(sub["trailing_vol_20"].median())
            vol_vol = float(sub["vol_of_vol_20"].median())
            dd = float(sub["drawdown"].median())
            trend = float(sub["trend_20"].median())
            cell_descriptors[cell] = np.array([
                regime_val,
                np.log(h),
                trail_vol * 100.0,
                vol_vol * 1000.0,
                dd,
                trend
            ], dtype=np.float32)
    return cell_descriptors


def extract_model_descriptor_matrix(
    dev_reference_df: pd.DataFrame | None = None
) -> dict[str, np.ndarray]:
    """Constructs model descriptor vector z_m combining static traits and anchor performance."""
    model_descriptors = {}
    for m, static in MODEL_STATIC_FEATURES.items():
        vec = [
            static["family_stat"],
            static["family_ml"],
            static["family_foundation"],
            static["log_param_count"] / 8.0,
            static["context_length"] / 256.0,
            static["is_pretrained"],
            np.log10(static["base_cost"] + 1e-4)
        ]
        # If anchor reference is available, compute anchor cell loss
        if dev_reference_df is not None:
            m_sub = dev_reference_df[dev_reference_df["model_id"] == m]
            calm_loss = m_sub[m_sub["regime"] == "calm"]["qlike_loss"].mean() if len(m_sub) else 0.5
            norm_loss = m_sub[m_sub["regime"] == "normal"]["qlike_loss"].mean() if len(m_sub) else 0.5
            vec.extend([float(calm_loss), float(norm_loss)])
        else:
            vec.extend([0.5, 0.5])
        model_descriptors[m] = np.array(vec, dtype=np.float32)
    return model_descriptors
