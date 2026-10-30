import gc
import json
import time
import traceback
import numpy as np
import pandas as pd
from .config import ROOT,save_json
from .preprocessing import process,split,FEATURES
from .regime_cells import pool,DESCRIPTORS,weights
from .forecasting_adapters import Adapter,NAMES,sync
from .model_descriptors import descriptors
from .performance_map import BayesianMap,design_matrix
from .evaluation import RestrictedEvaluator,balanced_truth
from .simulator import run,POLICIES,ABLATIONS

def evaluate(adapter,windows,data,fold,period):
    rows=[]
    for _,r in windows.iterrows():
        history=data[r.asset].loc[:r.date,FEATURES]
        sync(); tick=time.perf_counter()
        pred=adapter.predict(history,int(r.h))
        sync(); elapsed=time.perf_counter()-tick
        rows.append(dict(model=adapter.name,asset=r.asset,date=r.date,h=r.h,cell=int(r.cell),target=r.target,
            prediction=pred,loss=abs(np.log(max(r.target,1e-10)/pred)),cost=max(elapsed,1e-8),
            key=f'{fold}/{adapter.name}/{r.asset}/{r.date}/{r.h}',period=period))
    return pd.DataFrame(rows)

def reference(cfg):
    from .validation import validate
    validate(cfg)
    data=process(cfg); status=[]
    status_path=ROOT/f'results/manifests/model_status_{cfg["hash"]}.json'
    previous=json.loads(status_path.read_text()) if status_path.exists() else []
    for fold,(train_end,cal_end,test_end) in enumerate(cfg['folds'],1):
        train={t:split(d,2010,train_end) for t,d in data.items()}
        thresholds=np.quantile(pd.concat(list(train.values())).vol,[1/3,2/3])
        cp=pool(data,train_end+1,cal_end,thresholds,cfg['windows_per_asset_cell'])
        tp=pool(data,cal_end+1,test_end,thresholds,cfg['windows_per_asset_cell'])
        folder=ROOT/f'artifacts/oracle_reference/{cfg["hash"]}/fold{fold}'
        folder.mkdir(parents=True,exist_ok=True)
        cp.to_csv(folder/'cal_windows.csv',index=False);tp.to_csv(folder/'test_windows.csv',index=False)
        save_json(folder/'regimes.json',dict(thresholds=thresholds.tolist(),training_end=train_end,
                   minimum=cfg['minimum_cell_samples'],sampling='even spacing independently per asset/cell'))
        counts=tp.groupby(['cell','asset']).size().unstack(fill_value=0).reindex(range(12),fill_value=0)
        counts.to_csv(ROOT/f'results/tables/occupancy_fold{fold}.csv')
        if (counts.sum(1)<cfg['minimum_cell_samples']).any():
            raise ValueError(f'Sparse fold {fold}; predefined rule: halt fold, never merge cells')
        for name in NAMES:
            if name in ('chronos','timesfm') and not cfg['foundation_models']: continue
            if (folder/f'{name}_test.csv').exists():
                old=next((s for s in previous if s['fold']==fold and s['model']==name and s['status']=='complete'),None)
                status.append(old or dict(fold=fold,model=name,status='cached'));continue
            tick=time.perf_counter()
            try:
                print('FIT',fold,name,flush=True)
                a=Adapter(name,cfg['device'],cfg['context'],flip_invariance=cfg.get('timesfm_flip_invariance',False)).fit(train)
                # Warm-up excluded from per-origin cost and recorded separately.
                warm=time.perf_counter();a.predict(next(iter(train.values()))[FEATURES],5);sync()
                warm=time.perf_counter()-warm
                for period,p in [('cal',cp),('test',tp)]:
                    if (folder/f'{name}_{period}.csv').exists(): continue
                    out=evaluate(a,p,data,fold,period)
                    out.to_csv(folder/f'{name}_{period}.csv',index=False)
                memory=None
                if name in ('chronos','timesfm'):
                    import torch
                    memory=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
                status.append(dict(fold=fold,model=name,status='complete',wall_seconds=time.perf_counter()-tick,
                                   warmup_seconds=warm,peak_cuda_bytes=memory))
                del a;gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError: pass
            except Exception as exc:
                status.append(dict(fold=fold,model=name,status='failed',error=repr(exc),traceback=traceback.format_exc()))
            save_json(ROOT/'results/manifests/model_status.json',status)
            save_json(status_path,status)
    save_json(ROOT/'results/manifests/model_status.json',status)
    save_json(status_path,status)

def setup_map(cal,names,cost,windows,kind='full'):
    z,static,table=descriptors(names,cal,cost)
    x=windows.groupby('cell')[DESCRIPTORS].mean().reindex(range(12)).to_numpy()
    x=(x-x.mean(0))/np.maximum(x.std(0),1e-8)
    x=np.c_[x,np.array([c%2 for c in range(12)])]
    if kind=='no_behaviour': z=static
    if kind=='low_rank':
        mat=cal.groupby(['model','cell']).loss.mean().unstack().reindex(names).fillna(cal.loss.mean()).to_numpy()
        u,s,_=np.linalg.svd(mat-mat.mean(),full_matrices=False);z=u[:,:2]*s[:2]
    design=design_matrix(z,x,'independent' if kind=='independent_centered' else ('full' if kind in ('low_rank','no_behaviour') else kind))
    global_var=max(cal.loss.var(),.05)
    variances=cal.groupby('cell').loss.agg(['var','count']).reindex(range(12)).fillna(0)
    noise=(variances['var']*variances['count']+20*global_var)/(variances['count']+20)
    posterior=BayesianMap(design,np.tile(np.maximum(noise,.02),len(names)))
    if kind=='independent_centered':
        # Empirical-Bayes centring avoids shrinking positive normalised losses to zero.
        posterior.mean[:]=float(cal.loss.mean())
        posterior._pair_mean[:]=float(cal.loss.mean())
    for (m,c),g in cal.groupby(['model','cell']):
        posterior.update(names.index(m)*12+int(c),g.loss.mean(),max(.1,len(g)*.1))
    return posterior,table

def simulations(cfg):
    all_results=[];map_metrics=[];forecast=[];descs=[];generalisation=[];runtimes=[]
    for fold in range(1,len(cfg['folds'])+1):
        if cfg.get('_only_fold') is not None and fold!=cfg['_only_fold']:continue
        folder=ROOT/f'artifacts/oracle_reference/{cfg["hash"]}/fold{fold}'
        names=[n for n in NAMES if (folder/f'{n}_test.csv').exists()]
        if len(names)<2: raise RuntimeError('Not enough completed models')
        cal=pd.concat([pd.read_csv(folder/f'{n}_cal.csv') for n in names],ignore_index=True)
        test=pd.concat([pd.read_csv(folder/f'{n}_test.csv') for n in names],ignore_index=True)
        test.drop_duplicates(['asset','date','h']).groupby(['cell','asset']).size().unstack(fill_value=0).reindex(range(12),fill_value=0).to_csv(ROOT/f'results/tables/occupancy_fold{fold}.csv')
        scale=max(cal.loss.median(),.1)  # Calibration-only loss normalisation.
        cal.loss/=scale;test.loss/=scale
        for d in [cal,test]: d['pair']=d.model.map({n:i for i,n in enumerate(names)})*12+d.cell
        # Conservative measured calibration maxima; evaluator charges actual test timings.
        cost=cal.groupby('pair').cost.max().reindex(range(len(names)*12)).to_numpy()
        test['measured_cost']=test.cost
        for name,g in test.groupby('model'):
            runtimes.append(dict(fold=fold,model=name,evaluations=len(g),total_inference_seconds=g.cost.sum(),
                median_seconds=g.cost.median(),p95_seconds=g.cost.quantile(.95),minimum_seconds=g.cost.min()))
        windows=pd.read_csv(folder/'cal_windows.csv')
        full,table=setup_map(cal,names,cost,windows)
        table['fold']=fold;descs.append(table)
        epsilon=float(np.clip(cal.groupby(['model','cell']).loss.mean().groupby('cell').std().median()*.1,.01,.1))
        save_json(folder/'frozen_calibration.json',dict(scale=scale,epsilon=epsilon,confidence=cfg['confidence'],
                  cost=cost.tolist(),cost_rule='calibration maximum per model/cell',hash=cfg['hash']))
        # Chronological calibration holdout, no final-test inputs.
        dates=sorted(cal.date.unique());cut=dates[int(len(dates)*.7)]
        fitcal=cal[cal.date<cut];hold=cal[cal.date>=cut]
        for kind in ['global','independent','low_rank','no_behaviour','full','no_effects']:
            p,_=setup_map(fitcal,names,cost,windows,kind)
            mu,cov=p.predict();idx=hold.pair.to_numpy(int)
            sd=np.sqrt(np.maximum(np.diag(cov)[idx]+p.noise[idx],0))
            err=hold.loss.to_numpy()-mu[idx]
            map_metrics.append(dict(fold=fold,map=kind,rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err)),
                coverage95=np.mean(abs(err)<=1.96*sd),width95=np.mean(3.92*sd),n=len(hold)))
            if kind=='full':
                pd.DataFrame(dict(observed=hold.loss.to_numpy(),predicted=mu[idx],sd=sd,cell=hold.cell.to_numpy())).to_csv(
                    ROOT/f'results/tables/map_predictions_fold{fold}.csv',index=False)
        truth=balanced_truth(test,names)
        for axis,values in [('cell',range(12)),('model',names)]:
            for value in values:
                observed=cal[cal[axis]!=value];unseen=cal[cal[axis]==value]
                p,_=setup_map(observed,names,cost,windows)
                mu,cov=p.predict();ix=unseen.pair.to_numpy(int)
                error=unseen.loss.to_numpy()-mu[ix];sd=np.sqrt(np.maximum(np.diag(cov)[ix]+p.noise[ix],1e-10))
                generalisation.append(dict(fold=fold,held_out_axis=axis,held_out_value=value,n=len(unseen),
                    rmse=np.sqrt(np.mean(error**2)),coverage95=np.mean(abs(error)<=1.96*sd),exploratory=True,
                    missing_behaviour_handling='zero imputation before descriptor scaling'))
        forecast.append(test.groupby(['model','cell','h']).agg(loss=('loss','mean'),measured_seconds=('measured_cost','sum')).reset_index().assign(fold=fold))
        for policy in POLICIES+ABLATIONS:
            initial=full
            if policy in ('no_descriptors','no_behaviour'): initial,_=setup_map(cal,names,cost,windows,policy)
            for seed in cfg['policy_seeds']:
                print('POLICY',fold,policy,seed,flush=True)
                checkpoint=ROOT/f'artifacts/observed_losses/{cfg["hash"]}/fold{fold}_{policy}_{seed}.json'
                if checkpoint.exists():
                    try:saved=json.loads(checkpoint.read_text())
                    except json.JSONDecodeError:saved={}
                    if 'results' in saved:
                        for r in saved['results']:
                            chosen=r['selected'];w0=weights()
                            r['regret']=float(truth[chosen]@w0-(truth@w0).min())
                            r['equal_weight_regret']=float(truth[chosen].mean()-truth.mean(1).min())
                            r['agreement']=int(chosen==int((truth@w0).argmin()))
                        all_results.extend(saved['results']);continue
                ev=RestrictedEvaluator(test,len(cost),seed)
                cc=dict(cfg,_full_cost=test.cost.sum())
                w=weights(policy!='no_weight')
                results,trace=run(policy,ev,initial,cost,truth,w,cc,seed,epsilon)
                for r in results:r.update(fold=fold,scope=cfg.get('scope','medium'),models=','.join(names))
                all_results.extend(results)
                save_json(checkpoint,dict(trace=trace,revealed=ev.log,results=results))
        pd.DataFrame(all_results).to_csv(ROOT/'results/tables/policy_budget_results.csv',index=False)
    result=pd.DataFrame(all_results)
    pd.DataFrame(generalisation).to_csv(ROOT/'results/tables/generalisation_metrics.csv',index=False)
    pd.DataFrame(runtimes).to_csv(ROOT/'results/tables/model_runtime_summary.csv',index=False)
    abst=[]
    for (policy,budget),g in result.groupby(['policy','budget_fraction']):
        rec=g[g.abstain==0]
        abst.append(dict(policy=policy,budget_fraction=budget,recommendation_fraction=len(rec)/len(g),
            accuracy_among_recommended=rec.agreement.mean() if len(rec) else np.nan,
            confidently_incorrect_fraction=g.confidently_wrong.mean(),n=len(g),recommendations=len(rec)))
    pd.DataFrame(abst).to_csv(ROOT/'results/tables/abstention_summary.csv',index=False)
    for name,d in [('performance_map_metrics',pd.DataFrame(map_metrics)),('model_descriptors',pd.concat(descs)),
        ('model_forecasting_performance',pd.concat(forecast)),('decision_regret_by_budget',result),
        ('regime_wise_results',result[['fold','policy','seed','budget_fraction']+[f'cell_{c}_regret' for c in range(12)]]),
        ('ablation_results',result[result.policy.isin(ABLATIONS+['act','fixed'])]),
        ('robustness_results',result[result.policy.isin(['act','no_weight'])]),
        ('abstention_results',result),('runtime_and_cost_results',result)]:
        d.to_csv(ROOT/f'results/tables/{name}.csv',index=False)
