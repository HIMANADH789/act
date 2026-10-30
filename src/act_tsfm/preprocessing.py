import numpy as np
import pandas as pd
from .config import ROOT

FEATURES = ['v','mean5','mean20','mean60','ewma','lag1','lag2','trend','vol','vov','drawdown','liquidity']

def features(raw: pd.DataFrame) -> pd.DataFrame:
    p = raw['Adj Close'].where(raw['Adj Close']>0)
    r = np.log(p).diff()
    d = pd.DataFrame(index=raw.index)
    d['v'] = r**2
    for n in (5,20,60):
        d[f'mean{n}'] = d.v.rolling(n).mean()
    d['ewma'] = d.v.ewm(alpha=.06,adjust=False).mean()
    d['lag1'], d['lag2'] = d.v.shift(1), d.v.shift(2)
    d['trend'] = np.log(p/p.shift(20))
    d['vol'] = r.rolling(20).std()
    d['vov'] = d.vol.rolling(20).std()
    d['drawdown'] = p/p.rolling(60).max()-1
    d['liquidity'] = np.log((raw['Close']*raw['Volume']).clip(lower=1)).rolling(20).mean()
    # Future targets are isolated from FEATURES and never passed to adapters.
    for h in (1,5):
        d[f'rv{h}'] = sum(d.v.shift(-j) for j in range(1,h+1))
        d[f'outcome{h}'] = pd.Series(d.index,index=d.index).shift(-h)
    return d

def process(cfg: dict) -> dict[str,pd.DataFrame]:
    out = {}
    for t in cfg['tickers']:
        p = ROOT/f'data/raw/{t}.csv'
        if p.exists():
            raw = pd.read_csv(p,index_col=0,parse_dates=True).sort_index()
            if len(raw)<3500: continue
            if raw.index.duplicated().any(): raise ValueError(f'Duplicate dates {t}')
            out[t] = features(raw)
            out[t].to_csv(ROOT/f'data/processed/{t}.csv')
    return out

def split(d: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    # Drop five origins at each edge, plus require full target in period.
    q = d[(d.index.year>=start)&(d.index.year<=end)].iloc[5:-5]
    return q.dropna(subset=FEATURES+['rv1','rv5'])
