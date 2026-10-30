"""Read-only scientific audit of the frozen experiment; writes versioned diagnostics."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT,save_json
from .forecasting_adapters import NAMES
from .runner import setup_map
from .regime_cells import weights
from .simulator import POLICIES,ABLATIONS
from .commissioning_diagnostics import regret_contract,posterior_contracts,corrected_recommendation_flags

AUDIT=ROOT/'results/audit_v1'

def read_fold(cfg: dict,fold: int) -> tuple:
    folder=ROOT/f'artifacts/oracle_reference/{cfg["hash"]}/fold{fold}'
    cal=pd.concat([pd.read_csv(folder/f'{n}_cal.csv') for n in NAMES],ignore_index=True)
    test=pd.concat([pd.read_csv(folder/f'{n}_test.csv') for n in NAMES],ignore_index=True)
    scale=max(cal.loss.median(),.1)
    for d in (cal,test):
        d['pair']=d.model.map({n:i for i,n in enumerate(NAMES)})*12+d.cell
        d['normalised_loss']=d.loss/scale
    truth=np.empty((8,12))
    for m,n in enumerate(NAMES):
        for c in range(12):truth[m,c]=test[(test.model==n)&test.cell.eq(c)].groupby('asset').normalised_loss.mean().mean()
    assert np.isfinite(truth).all()
    return folder,cal,test,truth,float(scale)

def epsilon_from_calibration(cal: pd.DataFrame) -> float:
    return float(np.clip(cal.groupby(['model','cell']).normalised_loss.mean().groupby('cell').std().median()*.1,.01,.1))

def audit_ledgers(cfg: dict) -> dict:
    for p in ('tables','figures','manifests'):(AUDIT/p).mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(ROOT/'results/tables/policy_budget_results.csv')
    assert len(r)==3*14*10*5 and not r.duplicated(['fold','policy','seed','budget_fraction']).any()
    expected={(f,p,s,b) for f in (1,2,3) for p in POLICIES+ABLATIONS for s in cfg['policy_seeds'] for b in cfg['budgets']}
    assert set(map(tuple,r[['fold','policy','seed','budget_fraction']].to_numpy()))==expected
    ledger=[];feasible=[];flags=[];hashes=[];scale_rows=[]
    for fold in (1,2,3):
        folder,rawcal,test,truth,scale=read_fold(cfg,fold)
        epsilon=epsilon_from_calibration(rawcal)
        frozen=json.loads((folder/'frozen_calibration.json').read_text())
        assert math.isclose(scale,frozen['scale'],rel_tol=1e-12)
        assert math.isclose(epsilon,frozen['epsilon'],rel_tol=1e-12)
        np.testing.assert_allclose(truth,pd.read_csv(folder/'full_reference_loss_matrix.csv',index_col=0).to_numpy(),atol=1e-12)
        contract=regret_contract(truth,weights(),epsilon)
        raw_contract=regret_contract(truth*scale,weights(),epsilon*scale)
        assert contract['all_cell_single_model_feasible']==raw_contract['all_cell_single_model_feasible']
        counts=test[test.model==NAMES[0]].groupby('cell').size().reindex(range(12),fill_value=0)
        assert counts.min()>=cfg['minimum_cell_samples']
        common=set(map(tuple,test[test.model==NAMES[0]][['asset','date','h','cell']].to_numpy()))
        for n in NAMES:assert set(map(tuple,test[test.model==n][['asset','date','h','cell']].to_numpy()))==common
        feasible.append(dict(fold=fold,epsilon=epsilon,normalisation_scale=scale,
            global_model=NAMES[contract['weighted_global_model']],minimax_model=NAMES[contract['minimax_model']],
            minimum_worst_cell_regret=contract['minimum_worst_cell_regret'],
            chosen_worst_cell_regret=contract['chosen_worst_cell_regret'],
            frozen_single_model_all_cell_feasible=contract['all_cell_single_model_feasible'],
            weighted_global_contract_feasible=contract['global_weighted_contract_feasible'],
            routing_contract_feasible=contract['regime_specific_contract_feasible'],
            weighted_global_loss=contract['weighted_global_loss'],routing_oracle_loss=contract['regime_oracle_loss'],
            price_of_single_model=contract['price_of_single_model'],minimum_cell_windows=int(counts.min())))
        scale_rows.append(dict(fold=fold,scale=scale,epsilon_normalised=epsilon,epsilon_raw_equivalent=epsilon*scale,
            tolerance_source='calibration only: clip(.1 * median_cell SD_across_models(mean_loss), .01, .1)',
            actual_target_cells='all 12; positive stress weights; unweighted within-cell regret'))
        cal=rawcal.copy();cal.loss=cal.normalised_loss
        cost=cal.groupby('pair').cost.max().reindex(range(96)).to_numpy()
        priors={kind:setup_map(cal,NAMES,cost,pd.read_csv(folder/'cal_windows.csv'),kind)[0]
                for kind in ('full','no_descriptors','no_behaviour')}
        full=priors['full'];prior_sha=hashlib.sha256(full.predict()[0].tobytes()+full.predict()[1].tobytes()+full.noise.tobytes()).hexdigest()
        bykey=test.set_index('key');fullcost=float(test.cost.sum())
        for policy in POLICIES+ABLATIONS:
            for seed in cfg['policy_seeds']:
                path=ROOT/f'artifacts/observed_losses/{cfg["hash"]}/fold{fold}_{policy}_{seed}.json'
                payload=path.read_bytes();saved=json.loads(payload)
                hashes.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
                observed=saved['revealed'];trace=saved['trace'];keys=[o['key'] for o in observed]
                assert len(keys)==len(set(keys))
                cumulative=np.cumsum([o['cost'] for o in observed])
                pos=0
                for t in trace:
                    b=t['batch'];part=observed[pos:pos+b];pos+=b
                    assert t['observations']==[o['key'] for o in part]
                    assert all(o['pair']==t['pair'] for o in part)
                    assert math.isclose(t['cost'],float(cumulative[pos-1]),rel_tol=1e-8,abs_tol=1e-8)
                    assert len(t['posterior_mean'])==96 and np.isfinite(t['posterior_mean']).all()
                assert pos==len(observed)
                for o in observed:
                    ref=bykey.loc[o['key']]
                    assert math.isclose(o['cost'],ref.cost,rel_tol=1e-8,abs_tol=1e-10)
                    assert math.isclose(o['loss'],ref.normalised_loss,rel_tol=1e-8,abs_tol=1e-10)
                    expected_loss=abs(np.log(max(ref.target,1e-10)/ref.prediction))/scale
                    assert math.isclose(o['loss'],expected_loss,rel_tol=1e-8,abs_tol=1e-10)
                initial=copy.deepcopy(priors[policy if policy in ('no_descriptors','no_behaviour') else 'full'])
                if trace:
                    t=trace[0];part=observed[:t['batch']]
                    initial.update(t['pair'],float(np.mean([o['loss'] for o in part])),len(part))
                    np.testing.assert_allclose(initial.predict()[0],t['posterior_mean'],rtol=1e-7,atol=1e-8)
                for record in saved['results']:
                    n=record['evaluations'];spent=float(cumulative[n-1]) if n else 0.
                    assert math.isclose(spent,record['cost'],rel_tol=1e-8,abs_tol=1e-8)
                    assert spent<=record['budget_fraction']*fullcost+1e-6
                    selected=record['selected'];regret=truth[selected]@weights()-(truth@weights()).min()
                    worst=float((truth[selected]-truth.min(0)).max())
                    assert math.isclose(regret,record['regret'],abs_tol=1e-9)
                    assert math.isclose(worst,record['worst_regime_regret'],abs_tol=1e-9)
                    row=r[(r.fold==fold)&r.policy.eq(policy)&r.seed.eq(seed)&r.budget_fraction.eq(record['budget_fraction'])].iloc[0]
                    assert row.selected==selected and math.isclose(row.cost,spent,abs_tol=1e-8)
                    corrected=corrected_recommendation_flags(not record['abstain'],record['probability'],worst,epsilon,cfg['confidence'])
                    flags.append(dict(fold=fold,policy=policy,seed=seed,budget_fraction=record['budget_fraction'],
                        frozen_confidently_wrong=record['confidently_wrong'],**corrected))
                ledger.append(dict(fold=fold,policy=policy,seed=seed,observations=len(observed),actions=len(trace),
                    unique_observations=True,measured_costs_match=True,losses_match=True,budgets_pass=True,
                    initial_evidence_sha256=prior_sha if policy not in ('no_descriptors','no_behaviour') else 'intended descriptor ablation',
                    first_posterior_update_pass=True))
        print('AUDIT_LEDGER_FOLD',fold,'PASS',flush=True)
    pd.DataFrame(ledger).to_csv(AUDIT/'tables/ledger_audit.csv',index=False)
    pd.DataFrame(feasible).to_csv(AUDIT/'tables/stopping_feasibility.csv',index=False)
    pd.DataFrame(scale_rows).to_csv(AUDIT/'tables/loss_scale_audit.csv',index=False)
    pd.DataFrame(flags).to_csv(AUDIT/'tables/recommendation_flag_audit.csv',index=False)
    save_json(AUDIT/'manifests/frozen_checkpoint_checksums.json',hashes)
    numeric=[c for c in ('regret','worst_regime_regret','agreement','abstain','cost','evaluations','acquisition_seconds','map_update_seconds')]
    per_seed=r.groupby(['policy','budget_fraction','seed'])[numeric].mean().reset_index()
    summary=per_seed.groupby(['policy','budget_fraction'])[numeric].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary['n_policy_seeds']=10;summary['n_fixed_folds']=3;summary['analysis_id']='audit_v1'
    summary.to_csv(AUDIT/'tables/policy_seed_summary.csv',index=False)
    snapshot=ROOT/'runs/frozen_69aaa8be2df3/results/tables'
    unchanged={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((snapshot/p.name).read_bytes()).hexdigest()
               for p in (ROOT/'results/tables').glob('*.csv') if (snapshot/p.name).exists()}
    protected=['policy_budget_results.csv','decision_regret_by_budget.csv','regime_wise_results.csv',
               'ablation_results.csv','abstention_results.csv','runtime_and_cost_results.csv',
               'model_forecasting_performance.csv','performance_map_metrics.csv']
    assert all(unchanged[p] for p in protected)
    data_manifest=json.loads((ROOT/'data/manifests/download.json').read_text())
    for asset in data_manifest:
        assert hashlib.sha256((ROOT/f'data/raw/{asset["ticker"]}.csv').read_bytes()).hexdigest()==asset['sha256']
    result=dict(analysis_id='audit_v1',frozen_hash=cfg['hash'],model_folds=24,policy_trajectories=len(ledger),
        budget_records=len(r),policy_seeds=cfg['policy_seeds'],preserved_csv_checksums=unchanged,
        initial_evidence='identical calibration prior for all primary policies; intentional descriptor ablations differ',
        access_boundary='cooperative in-process reveal-only API, not an adversarial OS sandbox',
        known_reporting_errors=['original summary SD pooled fold/seed rows instead of averaging folds per seed',
            'frozen confidently_wrong treats forced no-abstention outputs as confident',
            'original block bootstrap imputes unsupported asset/cells using pooled means',
            'independent map zero-centred prior shrinks positive losses toward zero; separate corrected calibration-only comparison'],
        policy_and_threshold_changes=False)
    save_json(AUDIT/'manifests/audit_integrity.json',result)
    return result

def replay_stopping(cfg: dict) -> None:
    rows=[];plans=[]
    for fold in (1,2,3):
        folder,rawcal,test,truth,scale=read_fold(cfg,fold);epsilon=epsilon_from_calibration(rawcal)
        cal=rawcal.copy();cal.loss=cal.normalised_loss
        costs=cal.groupby('pair').cost.max().reindex(range(96)).to_numpy()
        initial,_=setup_map(cal,NAMES,costs,pd.read_csv(folder/'cal_windows.csv'))
        caltruth=np.array([[cal[cal.model.eq(n)&cal.cell.eq(c)].groupby('asset').loss.mean().mean() for c in range(12)] for n in NAMES])
        cal_global=int((caltruth@weights()).argmin());cal_route=caltruth.argmin(0)
        for c in range(12):plans.append(dict(fold=fold,cell=c,calibration_routing_model=NAMES[cal_route[c]],
            reference_routing_model=NAMES[int(truth[:,c].argmin())],
            calibration_global_model=NAMES[cal_global],
            calibration_routing_test_cell_regret=float(truth[cal_route[c],c]-truth[:,c].min()),
            calibration_global_test_cell_regret=float(truth[cal_global,c]-truth[:,c].min()),
            stage='exploratory after examined final periods; calibration-only plan, frozen epsilon unchanged'))
        for seed in cfg['policy_seeds']:
            saved=json.loads((ROOT/f'artifacts/observed_losses/{cfg["hash"]}/fold{fold}_act_{seed}.json').read_text())
            posterior=copy.deepcopy(initial);cursor=0
            for record in saved['results']:
                while cursor<record['evaluations']:
                    o=saved['revealed'][cursor];posterior.update(o['pair'],o['loss']);cursor+=1
                rng=np.random.default_rng(100000+fold*1000+seed*10+int(record['budget_fraction']*10))
                d=posterior.draws(rng,2048).reshape(2048,8,12)
                out=posterior_contracts(d,record['selected'],weights(),epsilon)
                global_rec=out['exploratory_global_weighted_probability']>=cfg['confidence']
                routing_rec=out['exploratory_routing_joint_probability']>=cfg['confidence']
                route=out.pop('routing');cellprob=out.pop('cell_probabilities')
                selected=record['selected'];regime_regret=truth[route,np.arange(12)]-truth.min(0)
                rows.append(dict(fold=fold,seed=seed,budget_fraction=record['budget_fraction'],
                    frozen_abstain=record['abstain'],frozen_probability=record['probability'],
                    exploratory_global_recommend=int(global_rec),exploratory_routing_recommend=int(routing_rec),
                    global_regret=record['regret'],routing_worst_cell_regret=float(regime_regret.max()),
                    routing_weighted_regret=float(regime_regret@weights()),
                    exploratory_global_correct=int(global_rec and record['regret']<=epsilon),
                    exploratory_routing_correct=int(routing_rec and regime_regret.max()<=epsilon),
                    routing_models=','.join(NAMES[i] for i in route),routing_cell_probabilities=json.dumps(cellprob.tolist()),
                    **out,analysis_id='audit_v1_exploratory'))
        print('AUDIT_STOPPING_REPLAY',fold,'PASS',flush=True)
    pd.DataFrame(rows).to_csv(AUDIT/'tables/alternative_contract_replay.csv',index=False)
    pd.DataFrame(plans).to_csv(AUDIT/'tables/calibration_routing_plan.csv',index=False)

def holm(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,float);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out

def bootstrap_audit(cfg: dict,replicates: int=4000) -> None:
    """Support-aware paired asset/time blocks, conditional on recorded choices.

    Missing asset/cells are not invented. Replicates missing any target cell are
    excluded jointly across folds, with the exclusion rate reported explicitly.
    This is a versioned correction to the earlier pooled-mean fallback analysis.
    """
    rng=np.random.default_rng(20261029);batch=replicates*2;assets=cfg['tickers']
    aidx=rng.integers(len(assets),size=(batch,len(assets)));truths={};original={};eps={}
    for fold in (1,2,3):
        _,cal,d,truth,_=read_fold(cfg,fold);original[fold]=truth;eps[fold]=epsilon_from_calibration(cal)
        d['block']=pd.to_datetime(d.date).dt.to_period('Q').astype(str)
        byasset=[]
        for asset in assets:
            g=d[d.asset.eq(asset)];blocks=sorted(g.block.unique());sums=[];counts=[]
            for block in blocks:
                q=g[g.block.eq(block)].groupby(['model','cell']).normalised_loss.agg(['sum','count']).reindex(
                    pd.MultiIndex.from_product([NAMES,range(12)])).fillna(0)
                sums.append(q['sum'].to_numpy().reshape(8,12));counts.append(q['count'].to_numpy().reshape(8,12))
            ix=rng.integers(len(blocks),size=(batch,len(blocks)))
            ss=np.array(sums)[ix].sum(1);nn=np.array(counts)[ix].sum(1)
            byasset.append(np.divide(ss,nn,out=np.full_like(ss,np.nan),where=nn>0))
        arr=np.stack(byasset,axis=1)[np.arange(batch)[:,None],aidx]
        support=np.isfinite(arr).sum(1)
        truths[fold]=np.divide(np.nansum(arr,axis=1),support,out=np.full((batch,8,12),np.nan),where=support>0)
    valid=np.ones(batch,dtype=bool)
    for t in truths.values():valid&=np.isfinite(t).all(axis=(1,2))
    ids=np.flatnonzero(valid)[:replicates]
    if len(ids)<replicates:raise RuntimeError(f'Only {len(ids)} complete block replicates; inspect sparsity, do not impute')
    truths={f:t[ids] for f,t in truths.items()}
    seed_ix=rng.integers(10,size=(replicates,10))
    (AUDIT/'diagnostics').mkdir(exist_ok=True)
    np.savez_compressed(AUDIT/'diagnostics/block_bootstrap_draws.npz',**{f'fold{f}':t for f,t in truths.items()},
                        seed_indices=seed_ix,asset_indices=aidx[ids])
    r=pd.read_csv(ROOT/'results/tables/policy_budget_results.csv');comparisons=[];feas=[]
    for fold,t in truths.items():
        cell_regret=t-t.min(1,keepdims=True)
        best_worst=cell_regret.max(2).min(1)
        price=(t@weights()).min(1)-t.min(1)@weights()
        feas.append(dict(fold=fold,epsilon=eps[fold],minimum_worst_cell_regret=regret_contract(original[fold],weights(),eps[fold])['minimum_worst_cell_regret'],
            ci_low=np.quantile(best_worst,.025),ci_high=np.quantile(best_worst,.975),
            resampled_infeasible_fraction=float((best_worst>eps[fold]).mean()),
            price_single_model_mean=float(price.mean()),price_ci_low=np.quantile(price,.025),price_ci_high=np.quantile(price,.975),
            replicates=replicates,interpretation='bootstrap stability of finite-reference estimates; not a posterior population probability'))
    baselines=['fixed','random','risk','uncertainty','ucb','racing']
    frozen_secondary=pd.read_csv(ROOT/'runs/frozen_69aaa8be2df3/results/tables/statistical_comparisons.csv')
    for budget,g in r.groupby('budget_fraction'):
        observed=g.groupby(['policy','seed']).regret.mean().unstack()
        boot_scores={}
        for policy in ['act']+baselines:
            byfold=[]
            for fold in (1,2,3):
                selected=g[g.policy.eq(policy)&g.fold.eq(fold)].set_index('seed').selected.reindex(cfg['policy_seeds']).to_numpy(int)
                score=truths[fold]@weights()
                byfold.append(score[:,selected]-score.min(1)[:,None])
            boot_scores[policy]=np.mean(byfold,axis=0)
        for baseline in baselines:
            diff=observed.loc['act'].to_numpy()-observed.loc[baseline].to_numpy()
            boot=np.take_along_axis(boot_scores['act']-boot_scores[baseline],seed_ix,axis=1).mean(1)
            delta=float(diff.mean());centered=boot-boot.mean()
            p=float((1+np.count_nonzero(abs(centered)>=abs(delta)))/(replicates+1))
            secondary=frozen_secondary[frozen_secondary.budget_fraction.eq(budget)&frozen_secondary.baseline.eq(baseline)].iloc[0]
            comparisons.append(dict(analysis_id='audit_v1_support_aware',budget_fraction=budget,baseline=baseline,
                act_mean=observed.loc['act'].mean(),act_seed_std=observed.loc['act'].std(),
                baseline_mean=observed.loc[baseline].mean(),baseline_seed_std=observed.loc[baseline].std(),
                mean_difference=delta,paired_seed_difference_std=diff.std(ddof=1),
                block_ci_low=np.quantile(boot,.025),block_ci_high=np.quantile(boot,.975),
                familywise_ci_low=np.quantile(boot,.05/(2*6)),familywise_ci_high=np.quantile(boot,1-.05/(2*6)),
                block_p_centered=p,secondary_wilcoxon_p=secondary.p_value,secondary_wilcoxon_holm_p=secondary.holm_p,
                replicates=replicates,n_policy_seeds=10,n_fixed_folds=3,
                scope='conditional on saved decisions; assets joint across folds; within-asset calendar-quarter blocks; complete-support replicates'))
    out=pd.DataFrame(comparisons);out['block_holm_p_six_baselines']=np.nan
    for _,g in out.groupby('budget_fraction'):out.loc[g.index,'block_holm_p_six_baselines']=holm(g.block_p_centered.to_numpy())
    out['block_holm_p_all_30']=holm(out.block_p_centered.to_numpy())
    out.to_csv(AUDIT/'tables/statistical_comparisons.csv',index=False)
    pd.DataFrame(feas).to_csv(AUDIT/'tables/feasibility_block_uncertainty.csv',index=False)
    save_json(AUDIT/'manifests/bootstrap.json',dict(seed=20261029,requested=replicates,generated=batch,
        complete_support_replicates=int(valid.sum()),excluded=int((~valid).sum()),used=len(ids),
        exclusion_fraction=float((~valid).mean()),no_missing_cell_imputation=True,
        limitations=['conditional on learned policies and fixed trained models','three test periods fixed, not independent training replicates',
                    'eight asset clusters; conditioning on complete support may affect tails','does not retrain or rerun acquisition inside bootstrap'],
        correction='Frozen bootstrap filled unsupported asset/cells with pooled means; audit omits unsupported asset contributions and rejects incomplete target-cell replicates'))

def corrected_map_diagnostic(cfg: dict) -> None:
    rows=[]
    for fold in (1,2,3):
        folder,raw,_,_,_=read_fold(cfg,fold);cal=raw.copy();cal.loss=cal.normalised_loss
        cost=cal.groupby('pair').cost.max().reindex(range(96)).to_numpy()
        dates=sorted(cal.date.unique());cut=dates[int(.7*len(dates))];fit=cal[cal.date<cut];hold=cal[cal.date>=cut]
        for kind in ('independent','independent_centered'):
            p,_=setup_map(fit,NAMES,cost,pd.read_csv(folder/'cal_windows.csv'),kind)
            mu,cov=p.predict();ix=hold.pair.to_numpy(int);sd=np.sqrt(np.diag(cov)[ix]+p.noise[ix]);err=hold.loss.to_numpy()-mu[ix]
            rows.append(dict(fold=fold,map=kind,rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err)),
                coverage95=np.mean(abs(err)<=1.96*sd),width95=np.mean(3.92*sd),
                analysis_id='audit_v1_map_prior_correction' if kind.endswith('centered') else 'frozen_definition_reproduced',
                calibration_only=True,forecast_or_policy_rerun=False))
    pd.DataFrame(rows).to_csv(AUDIT/'tables/corrected_map_prior_comparison.csv',index=False)
