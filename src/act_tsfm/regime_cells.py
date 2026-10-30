import numpy as np
import pandas as pd
from .preprocessing import split

DESCRIPTORS = ['vol','vov','drawdown','trend','liquidity']

def cell_id(row: pd.Series, h: int, thresholds: np.ndarray) -> int:
    return int(np.searchsorted(thresholds,row['vol']))*4 + int(row['trend']>=0)*2 + int(h==5)

def weights(stress: bool = True) -> np.ndarray:
    w = np.array([(2 if c//4==2 else 1)*(1.5 if (c%4)//2==0 else 1) if stress else 1 for c in range(12)])
    return w/w.sum()

def pool(data: dict, start: int, end: int, thresholds: np.ndarray, limit: int) -> pd.DataFrame:
    records = []
    for asset,d in data.items():
        q = split(d,start,end)
        for date,row in q.iterrows():
            for h in (1,5):
                records.append(dict(asset=asset,date=date,h=h,cell=cell_id(row,h,thresholds),
                                    target=row[f'rv{h}'],**{k:row[k] for k in DESCRIPTORS}))
    p = pd.DataFrame(records)
    # Fixed evenly spaced sampling per asset/cell, independent of outcomes.
    return pd.concat([g.iloc[np.unique(np.linspace(0,len(g)-1,min(limit,len(g))).astype(int))]
                      for _,g in p.groupby(['asset','cell'])]).reset_index(drop=True)
