"""Prespecified synthetic conditions; never substitute these for ETF results."""
import numpy as np
import pandas as pd
from .config import ROOT
from .evaluation import RestrictedEvaluator
from .performance_map import BayesianMap
from .simulator import run,POLICIES
from .regime_cells import weights

def synthetic(cfg):
    rng=np.random.default_rng(2026);rows=[];raw=[]
    for scenario in ['equal','stable','changing','noisy_stress','sharp_cost','sparse']:
        n=4;mean=np.ones((n,12))
        if scenario!='equal':mean+=np.arange(n)[:,None]*.12
        if scenario in ('changing','noisy_stress'):mean[:,6:]=mean[::-1,6:]
        noise=np.ones(n*12)*(.5 if scenario=='noisy_stress' else .04)
        costs=np.repeat([1,3,10,30] if scenario=='sharp_cost' else [1,1,1,1],12).astype(float)
        records=[]
        for p in range(n*12):
            for k in range(2 if scenario=='sparse' and p%12>=8 else 8):
                records.append(dict(pair=p,loss=float(mean.ravel()[p]+rng.normal(0,np.sqrt(noise[p]))),cost=costs[p],key=f'{scenario}/{p}/{k}'))
        d=pd.DataFrame(records);raw.append(d.assign(scenario=scenario))
        initial=BayesianMap(np.eye(n*12),noise,ridge=.1)
        for p in range(n*12): initial.update(p,float(mean.ravel()[p]+rng.normal(0,.1)),1)
        for policy in POLICIES:
            for seed in cfg['policy_seeds']:
                ev=RestrictedEvaluator(d,n*12,seed)
                result,_=run(policy,ev,initial,costs,mean,weights(),dict(cfg,_full_cost=d.cost.sum()),seed,.02)
                rows.extend([dict(x,scenario=scenario,data_type='synthetic') for x in result])
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/synthetic_robustness_results.csv',index=False)
    pd.concat(raw).to_csv(ROOT/'artifacts/oracle_reference/synthetic_reference.csv',index=False)
