"""VOLARE dataset preprocessing, partitioning, target generation and regime feature extraction."""

from __future__ import annotations
import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd

FLOOR = 1e-10

PANEL_ASSETS = [
    {"symbol": "AAPL", "asset_type": "stocks", "category": "Equities"},
    {"symbol": "MSFT", "asset_type": "stocks", "category": "Equities"},
    {"symbol": "JPM", "asset_type": "stocks", "category": "Equities"},
    {"symbol": "XOM", "asset_type": "stocks", "category": "Equities"},
    {"symbol": "ES", "asset_type": "futures", "category": "Futures_Index"},
    {"symbol": "CL", "asset_type": "futures", "category": "Futures_Commodity"},
    {"symbol": "GC", "asset_type": "futures", "category": "Futures_Commodity"},
    {"symbol": "EURUSD", "asset_type": "forex", "category": "Forex"},
    {"symbol": "GBPUSD", "asset_type": "forex", "category": "Forex"},
    {"symbol": "USDJPY", "asset_type": "forex", "category": "Forex"},
]

DEV_END = "2023-12-31"
COMM_START = "2024-01-01"
COMM_END = "2025-12-31"
HOLDOUT_START = "2026-01-01"
HOLDOUT_END = "2026-06-30"
PURGE_DAYS = 22
HORIZONS = [1, 5, 22]


def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_volare(raw_dir: str = "data/volare_raw") -> dict[str, pd.DataFrame]:
    raw_data = {}
    for item in PANEL_ASSETS:
        sym = item["symbol"]
        atype = item["asset_type"]
        path = os.path.join(raw_dir, f"{atype}_{sym}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raw data file: {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        # Ensure rv5 is positive
        df["rv5"] = df["rv5"].clip(lower=FLOOR)
        if "close_price" not in df.columns:
            df["close_price"] = 100.0
        df["ret"] = np.log(df["close_price"] / df["close_price"].shift(1)).fillna(0.0)
        raw_data[sym] = df
    return raw_data


def process_features_and_targets(raw_data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Computes past-only features, targets Y_{t,h}, and fits regime thresholds strictly on development period."""
    processed = {}
    
    # First pass: collect development trailing RV to compute global/asset tertile thresholds
    dev_trailing_rv = []
    for sym, df in raw_data.items():
        dev_mask = df["date"] <= DEV_END
        dev_df = df[dev_mask].copy()
        dev_trailing = dev_df["rv5"].rolling(20, min_periods=5).mean().dropna()
        dev_trailing_rv.extend(dev_trailing.tolist())
        
    q33 = float(np.percentile(dev_trailing_rv, 33.333))
    q67 = float(np.percentile(dev_trailing_rv, 66.667))
    thresholds = {"tertile_low": q33, "tertile_high": q67}
    
    for item in PANEL_ASSETS:
        sym = item["symbol"]
        df = raw_data[sym].copy()
        
        # Targets: sum of future 1, 5, 22 days rv5
        for h in HORIZONS:
            # Y_{t,h} = \sum_{j=1}^h RV_{t+j}^{(5)}
            # shift(-1) is t+1, rolling(h).sum() backwards
            rev = df["rv5"].iloc[::-1]
            roll_sum = rev.rolling(h, min_periods=h).sum().iloc[::-1].shift(-1)
            df[f"target_h{h}"] = roll_sum.clip(lower=FLOOR)
        
        # Descriptors (strictly past-only)
        df["trailing_vol_20"] = np.sqrt(df["rv5"].rolling(20, min_periods=5).mean().clip(lower=FLOOR))
        df["vol_of_vol_20"] = df["rv5"].rolling(20, min_periods=5).std().fillna(0.0)
        
        # Drawdown from peak close
        peak = df["close_price"].cummax()
        df["drawdown"] = ((df["close_price"] - peak) / peak).clip(upper=0.0)
        
        # Trend strength: 20-day return momentum
        df["trend_20"] = (df["close_price"] / df["close_price"].shift(20) - 1.0).fillna(0.0)
        
        # Liquidity proxy (volume normalized or 0)
        if "volume" in df.columns:
            df["liquidity"] = np.log1p(df["volume"].fillna(0.0))
        else:
            df["liquidity"] = 0.0
            
        # Recent variability: 5-day std of returns
        df["ret_vol_5"] = df["ret"].rolling(5, min_periods=3).std().fillna(0.0)
        
        # Regime classification: Calm, Normal, Stress based on dev thresholds
        # Using trailing 20-day mean of rv5
        trail_rv = df["rv5"].rolling(20, min_periods=5).mean()
        regime = pd.Series(index=df.index, dtype=object)
        regime[trail_rv <= q33] = "calm"
        regime[(trail_rv > q33) & (trail_rv <= q67)] = "normal"
        regime[trail_rv > q67] = "stress"
        regime = regime.fillna("normal")
        df["regime"] = regime
        
        # Tag partitions
        df["partition"] = "unknown"
        df.loc[df["date"] <= DEV_END, "partition"] = "development"
        df.loc[(df["date"] >= COMM_START) & (df["date"] <= COMM_END), "partition"] = "commissioning"
        df.loc[(df["date"] >= HOLDOUT_START) & (df["date"] <= HOLDOUT_END), "partition"] = "holdout"
        
        processed[sym] = df
        
    return processed, thresholds


def extract_evaluation_windows(
    processed: dict[str, pd.DataFrame], 
    partition_name: str,
    windows_per_asset_cell: int = 4,
    seed: int = 42
) -> pd.DataFrame:
    """Extracts valid, non-leaking evaluation windows for each asset, regime, and horizon.
    
    A window w is defined by (asset, origin_date, horizon, regime, descriptors, targets).
    Purges edges to avoid partition boundary leakage.
    """
    rng = np.random.RandomState(seed)
    records = []
    
    for item in PANEL_ASSETS:
        sym = item["symbol"]
        category = item["category"]
        df = processed[sym]
        
        sub = df[df["partition"] == partition_name].copy()
        if len(sub) == 0:
            continue
            
        # Remove PURGE_DAYS at end of partition so target h=22 does not leak past partition
        max_idx = len(sub) - PURGE_DAYS
        if max_idx <= 0:
            continue
        sub = sub.iloc[:max_idx]
        
        for h in HORIZONS:
            target_col = f"target_h{h}"
            valid = sub.dropna(subset=[target_col, "trailing_vol_20", "vol_of_vol_20", "drawdown", "trend_20"])
            
            for regime in ["calm", "normal", "stress"]:
                cell_df = valid[valid["regime"] == regime]
                if len(cell_df) == 0:
                    continue
                # Sample evenly or select up to windows_per_asset_cell
                n_take = min(windows_per_asset_cell, len(cell_df))
                indices = np.linspace(0, len(cell_df) - 1, n_take, dtype=int)
                selected = cell_df.iloc[indices]
                
                for _, row in selected.iterrows():
                    origin_date = row["date"].strftime("%Y-%m-%d")
                    target_end = (row["date"] + pd.Timedelta(days=int(h * 1.5))).strftime("%Y-%m-%d")
                    window_id = f"{sym}_{origin_date}_h{h}_{regime}"
                    records.append({
                        "window_id": window_id,
                        "asset_id": sym,
                        "category": category,
                        "forecast_origin": origin_date,
                        "target_end_date": target_end,
                        "horizon": h,
                        "regime": regime,
                        "cell_id": f"{regime}_h{h}",
                        "target_variance": float(row[target_col]),
                        "trailing_vol_20": float(row["trailing_vol_20"]),
                        "vol_of_vol_20": float(row["vol_of_vol_20"]),
                        "drawdown": float(row["drawdown"]),
                        "trend_20": float(row["trend_20"]),
                        "liquidity": float(row["liquidity"]),
                        "ret_vol_5": float(row["ret_vol_5"]),
                    })
                    
    res_df = pd.DataFrame(records)
    return res_df

