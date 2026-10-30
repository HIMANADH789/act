"""Additional audit scope: every policy's evidence prefix and asset-subset stability."""
from __future__ import annotations
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from .config import ROOT,save_json
from .final_audit import AUDIT,read_fold,epsilon_from_calibration
from .runner import setup_map
from .simulator import POLICIES,ABLATIONS
from .forecasting_adapters import NAMES
from .regime_cells import weights
from .commissioning_diagnostics import regret_contract,posterior_contracts

def replay_fold(cfg: dict,fold: int) -> int:
    folder,rawcal,_,_,_=read_fold(cfg,fold);epsilon=epsilon_from_calibration(rawcal)
    cal=rawcal.copy();cal.loss=cal.normalised_loss
    cost=cal.groupby('pair').cost.max().reindex(range(96)).to_numpy()
    priors={k:setup_map(cal,NAMES,cost,pd.read_csv(folder/'cal_windows.csv'),k)[0] for k in ('full','no_descriptors','no_behaviour')}
    rows=[]
    for policy in POLICIES+ABLATIONS:
        for seed in cfg['policy_seeds']:
            saved=json.loads((ROOT/f'artifacts/observed_losses/{cfg["hash"]}/fold{fold}_{policy}_{seed}.json').read_text())
            posterior=copy.deepcopy(priors[policy if policy in ('no_descriptors','no_behaviour') else 'full'])
            cursor=0;prefix=[];count=0;endmeans={}
            for t in saved['trace']:count+=t['batch'];endmeans[count]=t['posterior_mean']
            for record in saved['results']:
                target=record['evaluations']
                while cursor<target:
                    o=saved['revealed'][cursor];posterior.update(o['pair'],o['loss'])
                    prefix.append((o['key'],o['loss']));cursor+=1
                assert cursor==target
                if target:np.testing.assert_allclose(posterior.predict()[0],endmeans[target],rtol=1e-7,atol=1e-8)
                rng=np.random.default_rng(990000+fold*10000+seed*100+int(record['budget_fraction']*10))
                out=posterior_contracts(posterior.draws(rng,1024),record['selected'],weights(policy!='no_weight'),epsilon)
                probability=out['primary_selected_joint_probability'];gate=probability>=cfg['confidence']
                forced=policy=='no_abstention';replayed_recommend=gate or forced
                rows.append(dict(fold=fold,policy=policy,seed=seed,budget_fraction=record['budget_fraction'],
                    evidence_count=target,prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                    posterior_mean_matches_saved_trace=True,only_revealed_prefix_used=True,
                    frozen_joint_probability=record['probability'],replay_joint_probability=probability,
                    frozen_recommend=int(not record['abstain']),replay_recommend=int(replayed_recommend),
                    decision_agrees=bool(replayed_recommend==bool(not record['abstain'])),
                    intentional_forced_output=forced,draws=1024,epsilon=epsilon,
                    scope='posterior reconstruction and independent MC audit, not a bitwise original RNG replay'))
        print('ALL_POLICY_REPLAY',fold,policy,'complete',flush=True)
    pd.DataFrame(rows).to_csv(AUDIT/f'tables/all_policy_replay_fold{fold}.csv',index=False)
    return fold

def all_policy_replay(cfg: dict) -> None:
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(replay_fold,cfg,f) for f in (1,2,3)]
        for job in futures:job.result()
    d=pd.concat([pd.read_csv(AUDIT/f'tables/all_policy_replay_fold{f}.csv') for f in (1,2,3)],ignore_index=True)
    assert len(d)==2100
    d.to_csv(AUDIT/'tables/all_policy_stopping_replay.csv',index=False)
    save_json(AUDIT/'manifests/all_policy_replay.json',dict(rows=len(d),trajectories=420,
        all_prefixes_isolated=bool(d.only_revealed_prefix_used.all()),all_posterior_means_match=bool(d.posterior_mean_matches_saved_trace.all()),
        decision_disagreements=int((~d.decision_agrees).sum()),draws_per_state=1024,
        limitation='independent Monte Carlo gate replay; original RNG states were not persisted'))

def subset_stability(cfg: dict,replicates: int=2000) -> None:
    rows=[];metadata=[]
    for label,assets in [('first_six',cfg['tickers'][:6]),('last_six',cfg['tickers'][2:])]:
        rng=np.random.default_rng(741 if label=='first_six' else 742)
        batch=replicates*3;aidx=rng.integers(6,size=(batch,6));draws={};originals={};eps={};counts={}
        for fold in (1,2,3):
            _,cal,test,_,_=read_fold(cfg,fold);d=test[test.asset.isin(assets)].copy();eps[fold]=epsilon_from_calibration(cal)
            counts[fold]=int(d[d.model.eq(NAMES[0])].groupby('cell').size().reindex(range(12),fill_value=0).min())
            truth=np.array([[d[d.model.eq(n)&d.cell.eq(c)].groupby('asset').normalised_loss.mean().mean() for c in range(12)] for n in NAMES])
            originals[fold]=regret_contract(truth,weights(),eps[fold])
            if counts[fold]<cfg['minimum_cell_samples']:continue
            d['block']=pd.to_datetime(d.date).dt.to_period('Q').astype(str);aa=[]
            for asset in assets:
                g=d[d.asset.eq(asset)];blocks=sorted(g.block.unique());ss=[];nn=[]
                for b in blocks:
                    q=g[g.block.eq(b)].groupby(['model','cell']).normalised_loss.agg(['sum','count']).reindex(pd.MultiIndex.from_product([NAMES,range(12)])).fillna(0)
                    ss.append(q['sum'].to_numpy().reshape(8,12));nn.append(q['count'].to_numpy().reshape(8,12))
                ix=rng.integers(len(blocks),size=(batch,len(blocks)));s=np.array(ss)[ix].sum(1);n=np.array(nn)[ix].sum(1)
                aa.append(np.divide(s,n,out=np.full_like(s,np.nan),where=n>0))
            a=np.stack(aa,axis=1)[np.arange(batch)[:,None],aidx];n=np.isfinite(a).sum(1)
            draws[fold]=np.divide(np.nansum(a,axis=1),n,out=np.full((batch,8,12),np.nan),where=n>0)
        valid=np.ones(batch,bool)
        for d in draws.values():valid&=np.isfinite(d).all(axis=(1,2))
        ids=np.flatnonzero(valid)[:replicates]
        if draws and len(ids)<replicates:raise RuntimeError('Insufficient supported subset bootstrap replicates')
        for fold in (1,2,3):
            point=originals[fold];eligible=fold in draws
            if eligible:
                t=draws[fold][ids];v=(t-t.min(1,keepdims=True)).max(2).min(1)
                lo,hi=np.quantile(v,[.025,.975]);frequency=float((v>eps[fold]).mean())
            else:lo=hi=frequency=np.nan
            rows.append(dict(subset=label,assets=','.join(assets),fold=fold,epsilon=eps[fold],
                observed_minimum_worst_cell_regret=point['minimum_worst_cell_regret'],
                finite_reference_feasible=point['all_cell_single_model_feasible'],minimum_cell_windows=counts[fold],
                uncertainty_supported=eligible,ci_low=lo,ci_high=hi,resampled_infeasible_fraction=frequency,
                replicates=replicates if eligible else 0,
                scope='same frozen eight-asset calibration scale/epsilon; six-asset commissioning pool; post hoc evaluator diagnostic'))
        metadata.append(dict(subset=label,generated=batch,used=len(ids),excluded=int((~valid).sum()),
                             minimum_cell_rule=cfg['minimum_cell_samples'],imputation=False))
    pd.DataFrame(rows).to_csv(AUDIT/'tables/subset_feasibility_stability.csv',index=False)
    save_json(AUDIT/'manifests/subset_bootstrap.json',metadata)
