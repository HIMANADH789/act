import numpy as np

def decision(draws: np.ndarray,w: np.ndarray,epsilon: float,confidence: float=.95) -> dict:
    d=draws.reshape(len(draws),-1,12)
    selected=int((d.mean(0)@w).argmin())
    regret=d[:,selected,:]-d.min(axis=1)
    probs=(regret<=epsilon).mean(0)
    # Joint criterion is at least as strict as each required cell criterion.
    joint=float((regret<=epsilon).all(axis=1).mean())
    return dict(selected=selected,recommend=joint>=confidence,probability=joint,
                lacking_cells=np.flatnonzero(probs<confidence).tolist())
