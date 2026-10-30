"""Common positive variance adapter; foundation models are zero-shot on squared returns."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..preprocessing import FEATURES

FLOOR=1e-10
NAMES=['persistence','rolling','ewma','har','ridge','histgb','chronos','timesfm']

class Adapter:
    def __init__(self,name: str,device: str='cuda',context: int=256,seed: int=42,flip_invariance: bool=False):
        self.name,self.device,self.context,self.seed=name,device,context,seed
        self.flip_invariance=flip_invariance
        self.models={}

    def fit(self,train: dict) -> 'Adapter':
        if self.name in ('chronos','timesfm'):
            import torch
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32=True
        if self.name in ('har','ridge','histgb'):
            import pandas as pd
            q=pd.concat(list(train.values())).dropna(subset=FEATURES+['rv1','rv5'])
            self.cols=['v','mean5','mean20'] if self.name=='har' else FEATURES
            for h in (1,5):
                est=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,l2_regularization=10,random_state=self.seed) if self.name=='histgb' else make_pipeline(StandardScaler(),Ridge(alpha=20))
                self.models[h]=est.fit(q[self.cols],np.log(q[f'rv{h}'].clip(lower=FLOOR)))
        elif self.name=='chronos':
            import torch
            from chronos import ChronosPipeline
            torch.manual_seed(self.seed)
            self.net=ChronosPipeline.from_pretrained('amazon/chronos-t5-tiny',device_map=self.device,torch_dtype=torch.float32)
        elif self.name=='timesfm':
            import timesfm
            self.net=timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch',torch_compile=False)
            self.net.compile(timesfm.ForecastConfig(max_context=self.context,max_horizon=32,normalize_inputs=True,infer_is_positive=True,force_flip_invariance=self.flip_invariance))
        return self

    def predict(self,history, h: int) -> float:
        row=history.iloc[-1]
        if self.name in ('persistence','rolling','ewma'):
            y=h*row[{'persistence':'v','rolling':'mean20','ewma':'ewma'}[self.name]]
        elif self.name in ('har','ridge','histgb'):
            y=np.exp(self.models[h].predict(history[self.cols].iloc[[-1]])[0])
        else:
            import torch
            a=history.v.iloc[-self.context:].to_numpy(dtype=np.float32)
            if self.name=='chronos':
                with torch.inference_mode():
                    pred=self.net.predict(torch.tensor(a),prediction_length=h,num_samples=20)
                y=pred[0].float().mean(0).clamp_min(FLOOR).sum().item()
            else:
                pred,_=self.net.forecast(horizon=h,inputs=[a])
                y=np.maximum(pred[0,:h],FLOOR).sum()
        if not np.isfinite(y): raise ValueError(f'Nonfinite {self.name} prediction')
        return float(max(y,FLOOR))

def sync() -> None:
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.synchronize()
    except ImportError: pass
