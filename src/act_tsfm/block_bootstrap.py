"""Paired asset/quarter block-bootstrap, conditional on recorded policy decisions.

Inference pairing is preserved across models. Folds remain chronological strata; assets
are resampled jointly across folds to retain shared-asset dependence. This does not
retrain forecasting models or re-run acquisition inside bootstrap replicates.
"""
import numpy as np
import pandas as pd
from .config import ROOT
from .forecasting_adapters import NAMES
from .regime_cells import weights

def compare(cfg: dict,results: pd.DataFrame,replicates: int=1000) -> pd.DataFrame:
    rng=np.random.default_rng(42);truths={};assets=cfg['tickers'];aidx=rng.integers(len(assets),size=(replicates,len(assets)))
    fallbacks=0
    for fold in sorted(results.fold.unique()):
        folder=ROOT/f'artifacts/oracle_reference/{cfg["hash"]}/fold{fold}'
        names=[n for n in NAMES if (folder/f'{n}_test.csv').exists()]
        d=pd.concat([pd.read_csv(folder/f'{n}_test.csv') for n in names])
        scale=max(pd.concat([pd.read_csv(folder/f'{n}_cal.csv') for n in names]).loss.median(),.1)
        d.loss/=scale;d['block']=pd.to_datetime(d.date).dt.to_period('Q').astype(str)
        global_mean=d.groupby(['model','cell']).loss.mean().reindex(pd.MultiIndex.from_product([names,range(12)])).to_numpy().reshape(len(names),12)
        by_asset=[]
        for asset in assets:
            g=d[d.asset==asset];blocks=sorted(g.block.unique());tensor=[];counts=[]
            for block in blocks:
                q=g[g.block==block].groupby(['model','cell']).loss.agg(['sum','count']).reindex(pd.MultiIndex.from_product([names,range(12)])).fillna(0)
                tensor.append(q['sum'].to_numpy().reshape(len(names),12));counts.append(q['count'].to_numpy().reshape(len(names),12))
            ix=rng.integers(len(blocks),size=(replicates,len(blocks)))
            sums=np.array(tensor)[ix].sum(1);ns=np.array(counts)[ix].sum(1)
            fallbacks+=int((ns==0).sum())
            by_asset.append(np.divide(sums,ns,out=np.broadcast_to(global_mean,sums.shape).copy(),where=ns>0))
        aa=np.stack(by_asset,axis=1)
        truths[fold]=aa[np.arange(replicates)[:,None],aidx].mean(1)
    rows=[];w=weights()
    for budget,g in results.groupby('budget_fraction'):
        scores={}
        for policy in ['act','fixed','random','risk','uncertainty','ucb','racing']:
            per_fold=[]
            for fold,fg in g[g.policy==policy].groupby('fold'):
                choices=fg.set_index('seed').selected.reindex(cfg['policy_seeds']).to_numpy(int)
                t=truths[fold]@w
                regret=t[:,choices]-t.min(1)[:,None]
                per_fold.append(regret)
            scores[policy]=np.mean(per_fold,axis=0)
        seed_ix=rng.integers(len(cfg['policy_seeds']),size=(replicates,len(cfg['policy_seeds'])))
        for baseline in ['fixed','random','risk','uncertainty','ucb','racing']:
            delta=scores['act']-scores[baseline]
            boot=np.take_along_axis(delta,seed_ix,axis=1).mean(1)
            original=g.groupby('policy').regret.mean()
            rows.append(dict(budget_fraction=budget,baseline=baseline,mean_difference=original['act']-original[baseline],
                ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),replicates=replicates,
                unit='paired assets and within-asset calendar-quarter blocks; joint policy-seed resampling',
                missing_cell_fallback_entries=fallbacks,
                limitation='conditional on recorded policy decisions; missing resampled asset cells use original pooled cell mean; folds fixed'))
    return pd.DataFrame(rows)
