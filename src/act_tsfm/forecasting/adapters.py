"""Forecasting model adapters implementing a uniform interface for HAR, GARCH, Chronos, and TimesFM."""

from __future__ import annotations
import time
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch

FLOOR = 1e-10


def qlike_loss(y_true: float, y_pred: float) -> float:
    y = max(float(y_true), FLOOR)
    y_hat = max(float(y_pred), FLOOR)
    ratio = y / y_hat
    return float(ratio - np.log(ratio) - 1.0)


class BaseForecaster:
    def __init__(self, model_id: str, device: str = "cuda", seed: int = 42):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.model_version = "1.0"
        self.fit_time_seconds = 0.0

    def fit(self, train_histories: dict[str, pd.DataFrame]) -> BaseForecaster:
        raise NotImplementedError

    def predict(self, asset_id: str, history: pd.DataFrame, origin_date: str, horizon: int) -> tuple[float, float]:
        """Returns (predicted_variance, inference_time_seconds)."""
        raise NotImplementedError


class HARForecaster(BaseForecaster):
    def __init__(self, device: str = "cpu", seed: int = 42):
        super().__init__("har", device=device, seed=seed)
        self.model_version = "HAR-Corsi-2009-Ridge"
        self.models = {}  # (asset, h) -> Ridge model

    def fit(self, train_histories: dict[str, pd.DataFrame]) -> BaseForecaster:
        t0 = time.time()
        for sym, df in train_histories.items():
            df = df.copy()
            df["rv_d"] = df["rv5"]
            df["rv_w"] = df["rv5"].rolling(5, min_periods=5).mean()
            df["rv_m"] = df["rv5"].rolling(22, min_periods=22).mean()
            
            for h in [1, 5, 22]:
                target_col = f"target_h{h}"
                valid = df.dropna(subset=["rv_d", "rv_w", "rv_m", target_col])
                if len(valid) < 30:
                    continue
                X = valid[["rv_d", "rv_w", "rv_m"]].values
                y = valid[target_col].values
                reg = Ridge(alpha=1.0, positive=True)
                reg.fit(X, y)
                self.models[(sym, h)] = reg
        self.fit_time_seconds = time.time() - t0
        return self

    def predict(self, asset_id: str, history: pd.DataFrame, origin_date: str, horizon: int) -> tuple[float, float]:
        t0 = time.time()
        # Ensure past-only up to origin_date
        sub = history[history["date"] <= origin_date].copy()
        if len(sub) < 22:
            pred = max(float(sub["rv5"].iloc[-1] * horizon), FLOOR)
            return pred, time.time() - t0
        
        rv_d = sub["rv5"].iloc[-1]
        rv_w = sub["rv5"].iloc[-5:].mean()
        rv_m = sub["rv5"].iloc[-22:].mean()
        X = np.array([[rv_d, rv_w, rv_m]])
        
        model = self.models.get((asset_id, horizon))
        if model is None:
            pred = max(float(rv_d * horizon), FLOOR)
        else:
            pred = float(model.predict(X)[0])
            pred = max(pred, FLOOR)
        return pred, time.time() - t0


class GARCHForecaster(BaseForecaster):
    def __init__(self, device: str = "cpu", seed: int = 42):
        super().__init__("garch", device=device, seed=seed)
        self.model_version = "GARCH-1-1-Normal"
        self.fitted_params = {}

    def fit(self, train_histories: dict[str, pd.DataFrame]) -> BaseForecaster:
        t0 = time.time()
        from arch import arch_model
        for sym, df in train_histories.items():
            rets = df["ret"].dropna().values * 100.0  # scale percentage returns
            if len(rets) < 100:
                continue
            try:
                am = arch_model(rets, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
                res = am.fit(disp="off", show_warning=False)
                self.fitted_params[sym] = res
            except Exception:
                pass
        self.fit_time_seconds = time.time() - t0
        return self

    def predict(self, asset_id: str, history: pd.DataFrame, origin_date: str, horizon: int) -> tuple[float, float]:
        t0 = time.time()
        sub = history[history["date"] <= origin_date]
        res = self.fitted_params.get(asset_id)
        if res is None or len(sub) < 30:
            pred = max(float(sub["rv5"].iloc[-1] * horizon), FLOOR)
            return pred, time.time() - t0
        
        try:
            # Forecast horizon h ahead
            from arch import arch_model
            rets = sub["ret"].iloc[-250:].values * 100.0
            am = arch_model(rets, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
            res_sub = am.fit(disp="off", show_warning=False, starting_values=res.params)
            fc = res_sub.forecast(horizon=horizon)
            # Daily variance forecasts in %^2, convert back to decimal variance
            var_daily = fc.variance.values[-1, :horizon] / 10000.0
            pred = float(np.sum(var_daily))
            pred = max(pred, FLOOR)
        except Exception:
            pred = max(float(sub["rv5"].iloc[-1] * horizon), FLOOR)
            
        return pred, time.time() - t0


class ChronosForecaster(BaseForecaster):
    def __init__(self, device: str = "cuda", context: int = 256, seed: int = 42):
        super().__init__("chronos", device=device, seed=seed)
        self.model_version = "amazon/chronos-t5-tiny"
        self.context = context
        self.pipeline = None

    def fit(self, train_histories: dict[str, pd.DataFrame]) -> BaseForecaster:
        t0 = time.time()
        from chronos import ChronosPipeline
        torch.manual_seed(self.seed)
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_version,
            device_map=self.device,
            torch_dtype=torch.float32
        )
        self.fit_time_seconds = time.time() - t0
        return self

    def predict(self, asset_id: str, history: pd.DataFrame, origin_date: str, horizon: int) -> tuple[float, float]:
        t0 = time.time()
        sub = history[history["date"] <= origin_date]
        ctx_series = sub["rv5"].iloc[-self.context:].values.astype(np.float32)
        
        with torch.inference_mode():
            t_input = torch.tensor(ctx_series)
            forecast = self.pipeline.predict(t_input, prediction_length=horizon, num_samples=16)
            # forecast is [num_samples, horizon]
            mean_curve = forecast[0].float().mean(dim=0).cpu().numpy()
            pred = float(np.sum(np.clip(mean_curve, FLOOR, None)))
            pred = max(pred, FLOOR)
            
        return pred, time.time() - t0


class TimesFMForecaster(BaseForecaster):
    def __init__(self, device: str = "cuda", context: int = 256, seed: int = 42):
        super().__init__("timesfm", device=device, seed=seed)
        self.model_version = "google/timesfm-2.5-200m-pytorch"
        self.context = context
        self.model = None

    def fit(self, train_histories: dict[str, pd.DataFrame]) -> BaseForecaster:
        t0 = time.time()
        import timesfm
        torch.manual_seed(self.seed)
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.model_version,
            torch_compile=False
        )
        self.model.compile(timesfm.ForecastConfig(
            max_context=self.context,
            max_horizon=32,
            normalize_inputs=True,
            infer_is_positive=True,
            force_flip_invariance=False
        ))
        self.fit_time_seconds = time.time() - t0
        return self

    def predict(self, asset_id: str, history: pd.DataFrame, origin_date: str, horizon: int) -> tuple[float, float]:
        t0 = time.time()
        sub = history[history["date"] <= origin_date]
        ctx_series = sub["rv5"].iloc[-self.context:].values.astype(np.float32)
        
        pred_curve, _ = self.model.forecast(horizon=horizon, inputs=[ctx_series])
        pred_h = pred_curve[0, :horizon]
        pred = float(np.sum(np.clip(pred_h, FLOOR, None)))
        pred = max(pred, FLOOR)
        return pred, time.time() - t0


def build_forecasters(device: str = "cuda", seed: int = 42) -> dict[str, BaseForecaster]:
    return {
        "har": HARForecaster(device="cpu", seed=seed),
        "garch": GARCHForecaster(device="cpu", seed=seed),
        "chronos": ChronosForecaster(device=device, seed=seed),
        "timesfm": TimesFMForecaster(device=device, seed=seed),
    }

