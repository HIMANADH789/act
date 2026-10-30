"""Evaluator boundary. Policies receive reveal-only capabilities, no result table."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Action:
    pair: int
    size: int

class RestrictedEvaluator:
    def __init__(self,records: pd.DataFrame, n_pairs: int,seed: int):
        rng=np.random.default_rng(seed)
        self.__queues={p:g.iloc[rng.permutation(len(g))].to_dict('records') for p,g in records.groupby('pair')}
        self.__cursor=np.zeros(n_pairs,dtype=int)
        self.log=[]

    def counts(self) -> np.ndarray:
        return np.array([len(self.__queues.get(p,[]))-self.__cursor[p] for p in range(len(self.__cursor))])

    def quote(self, action: Action) -> float:
        """Evaluator-only timing metadata; never expose hidden loss or forecast values."""
        p,b=action.pair,action.size
        k=self.__cursor[p]
        return sum(r['cost'] for r in self.__queues.get(p,[])[k:k+b])

    def reveal(self,action: Action) -> list[dict]:
        p,b=action.pair,action.size
        if b<1 or b>self.counts()[p]: raise ValueError('Duplicate/exhausted or invalid observation request')
        k=self.__cursor[p]
        out=self.__queues[p][k:k+b]
        self.__cursor[p]+=b
        self.log.extend(out)
        return out

def balanced_truth(records: pd.DataFrame,names: list[str]) -> np.ndarray:
    # Asset means receive equal weight within each cell.
    return records.groupby(['model','cell','asset']).loss.mean().groupby(['model','cell']).mean().reindex(
        pd.MultiIndex.from_product([names,range(12)])).to_numpy().reshape(len(names),12)
