"""Versioned audit report and figures. Frozen observations and thresholds are untouched."""
from __future__ import annotations
import hashlib
import json
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT,save_json
from .final_audit import AUDIT,read_fold
from .regime_cells import weights
from .forecasting_adapters import NAMES

PRIMARY=['act','fixed','random','risk','uncertainty','ucb','racing']
FINAL=ROOT/'results/final_audit'

def table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v,(float,np.floating)):
            if not np.isfinite(v):return 'undefined'
            return '0' if abs(v)<1e-12 else f'{v:.5g}'
        return str(v).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join(
        '| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def generate(cfg: dict) -> None:
    status=json.loads((AUDIT/'manifests/status.json').read_text());assert status['status']=='complete'
    assert (AUDIT/'tables/all_policy_stopping_replay.csv').exists()
    assert (AUDIT/'tables/subset_feasibility_stability.csv').exists()
    shutil.copytree(AUDIT,FINAL,dirs_exist_ok=True)
    tab=ROOT/'results/tables';at=FINAL/'tables';fig=FINAL/'figures'
    for name in ['policy_budget_results','decision_regret_by_budget','regime_wise_results','ablation_results',
                 'robustness_results','abstention_results','runtime_and_cost_results']:
        shutil.copyfile(tab/f'{name}.csv',at/f'frozen_{name}.csv')
    r=pd.read_csv(tab/'policy_budget_results.csv');s=pd.read_csv(at/'policy_seed_summary.csv')
    comparisons=pd.read_csv(at/'statistical_comparisons.csv')
    feasibility=pd.read_csv(at/'stopping_feasibility.csv');uncert=pd.read_csv(at/'feasibility_block_uncertainty.csv')
    replay=pd.read_csv(at/'alternative_contract_replay.csv');plan=pd.read_csv(at/'calibration_routing_plan.csv')
    online=pd.read_csv(tab/'online_runtime_results.csv');prior=pd.read_csv(at/'corrected_map_prior_comparison.csv')
    data=pd.DataFrame(json.loads((ROOT/'data/manifests/download.json').read_text()))
    bootstrap=json.loads((AUDIT/'manifests/bootstrap.json').read_text())
    integrity=json.loads((AUDIT/'manifests/audit_integrity.json').read_text())
    flags=pd.read_csv(at/'recommendation_flag_audit.csv')
    within=r.groupby(['fold','policy','budget_fraction']).regret.agg(['mean','std','count']).reset_index()
    within=within.rename(columns={'mean':'seed_mean','std':'seed_std','count':'n_policy_seeds'})
    within.to_csv(at/'within_fold_seed_variability.csv',index=False)
    fold_dispersion=within.groupby(['policy','budget_fraction']).seed_mean.agg(['mean','std','min','max']).reset_index()
    fold_dispersion=fold_dispersion.rename(columns={'std':'fixed_fold_mean_sd','min':'minimum_fold_mean','max':'maximum_fold_mean'})
    fold_dispersion['interpretation']='descriptive dispersion of three fixed fold means; not independent market replicates'
    fold_dispersion.to_csv(at/'between_fold_variability.csv',index=False)
    all_replay=pd.read_csv(at/'all_policy_stopping_replay.csv')
    subset_feas=pd.read_csv(at/'subset_feasibility_stability.csv')
    replay_manifest=json.loads((FINAL/'manifests/all_policy_replay.json').read_text())
    # Nonsmooth minimax statistics can be upward-biased under resampling. Add a
    # conservative simultaneous paired-difference sensitivity, not a population claim.
    bootdraw=np.load(FINAL/'diagnostics/block_bootstrap_draws.npz');uniform=[]
    for fold in (1,2,3):
        _,_,_,point,_=read_fold(cfg,fold);draws=bootdraw[f'fold{fold}']
        q=float(np.quantile(np.ptp(draws-point[None,:,:],axis=1).max(1),.95))
        row=feasibility[feasibility.fold.eq(fold)].iloc[0];lower=max(0,row.minimum_worst_cell_regret-q)
        uniform.append(dict(fold=fold,uniform_pairwise_error_radius=q,minimum_worst_cell_regret=row.minimum_worst_cell_regret,
            conservative_lower=lower,conservative_upper=row.minimum_worst_cell_regret+q,epsilon=row.epsilon,
            lower_exceeds_epsilon=bool(lower>row.epsilon),
            interpretation='simultaneous bootstrap sensitivity; no finite-sample population guarantee'))
    uniform=pd.DataFrame(uniform);uniform.to_csv(at/'simultaneous_feasibility_sensitivity.csv',index=False)
    joined=r.merge(flags,on=['fold','policy','seed','budget_fraction'],validate='one_to_one')
    abst=[]
    for (policy,budget),g in joined.groupby(['policy','budget_fraction']):
        rec=g[g.abstain.eq(0)]
        abst.append(dict(policy=policy,budget_fraction=budget,coverage=1-g.abstain.mean(),abstention_rate=g.abstain.mean(),
            oracle_agreement_when_recommended=rec.agreement.mean() if len(rec) else np.nan,
            unsafe_all_cell_recommendation_rate=g.incorrect_recommendation.mean(),
            confidently_incorrect_rate=g.confidently_incorrect.mean(),recommendations=len(rec),cases=len(g)))
    abst=pd.DataFrame(abst);abst.to_csv(at/'abstention_summary.csv',index=False)
    # All corrected derivatives stay here. Original CSVs/manifests remain unchanged.
    s.to_csv(at/'summary.csv',index=False)
    ranks=[];cellrows=[];truths={}
    for fold in (1,2,3):
        _,_,d,t,_=read_fold(cfg,fold);truths[fold]=t
        for m,n in enumerate(NAMES):
            for c in range(12):
                cellrows.append(dict(fold=fold,model=n,cell=c,volatility=['low','medium','high'][c//4],
                    trend='negative' if (c%4)//2==0 else 'non-negative',horizon=5 if c%2 else 1,
                    normalised_loss=t[m,c],cell_regret=t[m,c]-t[:,c].min(),weight=weights()[c]))
            ranks.append(dict(fold=fold,model=n,weighted_loss=t[m]@weights(),
                weighted_global_regret=t[m]@weights()-(t@weights()).min(),
                worst_cell_regret=(t[m]-t.min(0)).max()))
    pd.DataFrame(ranks).to_csv(at/'global_model_ranking.csv',index=False)
    pd.DataFrame(cellrows).to_csv(at/'regime_model_performance.csv',index=False)
    plt.rcParams.update({'figure.dpi':160,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    files=[]
    def save(f,name):
        f.tight_layout();p=fig/(name+'.png');f.savefig(p);plt.close(f);files.append(p)
    f,ax=plt.subplots(figsize=(8,4.8))
    for p,g in s[s.policy.isin(PRIMARY)].groupby('policy'):
        g=g.sort_values('budget_fraction');mean=g.regret_mean.clip(lower=0)
        ax.plot(g.budget_fraction,mean,'o-',label=p)
        ax.fill_between(g.budget_fraction,np.maximum(0,mean-g.regret_std),mean+g.regret_std,alpha=.1)
    ax.set(xlabel='Fraction of full-reference inference cost',ylabel='Decision regret',title='Audit v1: fold-averaged means ± SD across 10 policy seeds')
    ax.legend(ncol=4,fontsize=8);save(f,'decision_regret_vs_budget')
    f,ax=plt.subplots(figsize=(8,4.8))
    for p,g in s[s.policy.isin(PRIMARY)].groupby('policy'):
        g=g.sort_values('budget_fraction');ax.errorbar(g.cost_mean,g.regret_mean.clip(lower=0),yerr=g.regret_std,marker='o',capsize=2,label=p)
    ax.set(xlabel='Mean charged inference seconds (folds averaged)',ylabel='Decision regret',title='Audit v1: measured reference charges; simulated revelation')
    ax.legend(ncol=4,fontsize=8);save(f,'decision_regret_vs_measured_cost')
    f,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    vmax=max(t.max() for t in truths.values())
    for fold,ax in enumerate(axes,1):
        im=ax.imshow(truths[fold],aspect='auto',vmin=0,vmax=vmax,cmap='viridis')
        ax.set_yticks(range(8),NAMES);ax.set_title(f'Fold {fold}: empirical normalised loss')
        ax.set_xticks(range(12));ax.set_ylabel('Candidate model')
    axes[-1].set_xlabel('Cell: volatility × trend × horizon (even 1-day, odd 5-day)')
    f.colorbar(im,ax=axes.tolist(),label='Normalised absolute log variance error',fraction=.025)
    f.subplots_adjust(right=.88,hspace=.35);p=fig/'regime_specific_model_performance.png';f.savefig(p,bbox_inches='tight');plt.close(f);files.append(p)
    f,ax=plt.subplots(figsize=(8,4.5));x=np.arange(3)
    ax.bar(x-.22,feasibility.epsilon,width=.22,label='Frozen epsilon')
    ax.bar(x,feasibility.minimum_worst_cell_regret,width=.22,label='Best achievable single-model worst-cell regret')
    ax.errorbar(x,feasibility.minimum_worst_cell_regret,
        yerr=np.vstack([np.maximum(0,feasibility.minimum_worst_cell_regret-uncert.ci_low),np.maximum(0,uncert.ci_high-feasibility.minimum_worst_cell_regret)]),fmt='none',color='black',capsize=4)
    ax.bar(x+.22,feasibility.chosen_worst_cell_regret,width=.22,label='Weighted oracle worst-cell regret')
    ax.set_xticks(x,['2019–2020','2021–2022','2023–2025']);ax.set(ylabel='Normalised regret',title='Audit v1: finite-reference stopping feasibility (block intervals)')
    ax.legend(fontsize=8);save(f,'full_information_regret_feasibility')
    f,ax=plt.subplots(figsize=(8,4.5))
    for name,column in [('Frozen all-cell global model','frozen_abstain'),('Exploratory weighted global','exploratory_global_recommend'),('Exploratory regime routing','exploratory_routing_recommend')]:
        g=replay.groupby('budget_fraction')[column].mean();coverage=1-g if column=='frozen_abstain' else g
        ax.plot(g.index,coverage,'o-',label=name)
    ax.set(xlabel='Inference-cost budget fraction',ylabel='Recommendation fraction',ylim=(-.03,1.03),title='Frozen and exploratory confidence contracts; epsilon unchanged')
    ax.legend(fontsize=8);save(f,'exploratory_contract_coverage')
    f,axes=plt.subplots(1,2,figsize=(10,4))
    for name,g in abst[abst.policy.isin(PRIMARY)].groupby('policy'):
        axes[0].plot(g.budget_fraction,g.abstention_rate,'o-',label=name)
    axes[0].set(xlabel='Budget fraction',ylabel='Abstention rate',ylim=(0,1.05),title='Frozen primary policies')
    axes[0].legend(fontsize=7,ncol=2)
    axes[1].axis('off');axes[1].text(.5,.55,'0 recommendations from primary policies\n\nConditional recommendation accuracy\nis undefined, not 100%.',ha='center',va='center')
    save(f,'abstention_rate_and_decision_quality')
    matrix=s[s.policy.isin(PRIMARY)].pivot(index='policy',columns='budget_fraction',values='regret_mean')
    f,ax=plt.subplots(figsize=(8,4));im=ax.imshow(matrix,aspect='auto',cmap='magma_r')
    ax.set_yticks(range(len(matrix)),matrix.index);ax.set_xticks(range(5),['10%','20%','40%','60%','80%'])
    for i in range(len(matrix)):
        for j in range(5):ax.text(j,i,f'{max(0,matrix.iloc[i,j]):.3f}',ha='center',va='center',fontsize=8)
    ax.set(xlabel='Matched inference-cost cap',title='Audit v1: acquisition-policy mean regret');f.colorbar(im,ax=ax,label='Regret');save(f,'acquisition_policy_comparisons')
    selected=['act','no_decision','no_cost','no_descriptors','no_behaviour','no_uncertainty','no_weight','no_abstention','fixed']
    matrix=s[s.policy.isin(selected)].pivot(index='policy',columns='budget_fraction',values='regret_mean').reindex(selected)
    f,ax=plt.subplots(figsize=(8,5));im=ax.imshow(matrix,aspect='auto',cmap='magma_r')
    ax.set_yticks(range(len(matrix)),matrix.index);ax.set_xticks(range(5),['10%','20%','40%','60%','80%'])
    ax.set(xlabel='Budget fraction',title='Frozen ablations: same stress-weighted regret endpoint');f.colorbar(im,ax=ax,label='Regret');save(f,'main_ablation_results')
    f,ax=plt.subplots(figsize=(9,4.5));bottom=np.zeros(3)
    components=['inference_seconds','preprocessing_seconds','acquisition_seconds','map_update_seconds']
    for c in components:
        ax.bar(online.policy,online[c],bottom=bottom,label=c.replace('_seconds',''));bottom+=online[c]
    remainder=np.maximum(0,online.wall_seconds-bottom);ax.bar(online.policy,remainder,bottom=bottom,label='other run overhead')
    for i,row in enumerate(online.itertuples()):ax.text(i,row.wall_seconds+.08,f'{row.wall_seconds:.2f}s; {row.evaluation_calls} calls',ha='center',fontsize=9)
    ax.set(ylabel='Policy-specific wall seconds',title=f'Online execution; shared model setup {online.setup_shared_seconds.iloc[0]:.2f}s separately')
    ax.legend(fontsize=8,ncol=3);save(f,'online_execution_cost_breakdown')
    f,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    ax.bar(x-.15,feasibility.weighted_global_loss,.3,label='One-model full-information optimum')
    ax.bar(x+.15,feasibility.routing_oracle_loss,.3,label='Regime-routing full-information optimum')
    ax.set_xticks(x,['Fold 1','Fold 2','Fold 3']);ax.set(ylabel='Stress-weighted normalised loss',title='Exploratory action-class comparison; different deployment contracts')
    ax.legend(fontsize=8);save(f,'global_vs_regime_specific_commissioning')
    main=s[s.policy.isin(PRIMARY)].copy()
    main['mean_sd']=main.apply(lambda q:f'{max(0,q.regret_mean):.5f} ± {q.regret_std:.5f}',axis=1)
    main_table=main.pivot(index='budget_fraction',columns='policy',values='mean_sd').reset_index()
    feasible_table=feasibility[['fold','global_model','minimax_model','epsilon','minimum_worst_cell_regret','chosen_worst_cell_regret','price_of_single_model']]
    alternative=replay.groupby('budget_fraction')[['exploratory_global_recommend','exploratory_routing_recommend','frozen_abstain']].mean().reset_index()
    alternative.to_csv(at/'alternative_coverage_summary.csv',index=False)
    subplot=pd.read_csv(tab/'asset_subset_robustness.csv').groupby(['subset','budget_fraction','policy']).regret.mean().unstack().reset_index()
    synthetic=pd.read_csv(tab/'synthetic_robustness_results.csv')
    syn=synthetic[synthetic.budget_fraction.eq(.2)].groupby(['scenario','policy']).regret.mean().unstack().reset_index()
    prior_summary=prior.groupby('map')[['rmse','mae','coverage95','width95']].mean().reset_index()
    map_metrics=pd.read_csv(tab/'performance_map_metrics.csv').groupby('map')[['rmse','coverage95','width95']].mean().reset_index()
    conclusion='B and D: promising numerical trends, inconclusive broad superiority, and an empirically infeasible single-model/all-cell stopping contract.'
    lines=['# ACT-TSFM — audited final experiment report',
        f'Analysis version: **audit_v1**. Frozen primary configuration: `{cfg["hash"]}`. Original outputs are preserved in `../runs/frozen_69aaa8be2df3/results/`. No policy outcomes, forecasts, regime weights or epsilon thresholds were changed by this audit.',
        '## 1. Abstract',
        'The medium experiment completed three chronological folds, eight forecasters, twelve cells, ten policy seeds and five cost budgets. ACT has lower mean provisional decision regret than fixed and random selection at all budgets, but adaptive-baseline comparisons are mixed and uncertainty prevents a general superiority claim. All primary policies abstain. Evaluator-side analysis shows that no single candidate meets the frozen all-cell tolerance in the finite reference; this is a different obstacle from not knowing the best weighted global model. The online check establishes execution, not practical compute savings.',
        '## 2. Problem definition',
        'The primary decision is a single model minimising stress-weighted expected cell loss. The stopping contract additionally demands that this same model be epsilon-optimal relative to the best candidate in every cell. These objectives are not equivalent. Provisional model-choice regret is measurable even when commissioning is refused; it must not be described as recommendation accuracy.',
        '## 3. Method and Bayesian performance map',
        'A Gaussian Bayesian linear map uses static/anchor behavioural descriptors, cell descriptors, bilinear interactions and regularised model/cell effects. Calibration cell noise is partially pooled. Calibration counts are discounted by 0.1; coefficient prior precision is 5. Posterior mean-loss uncertainty is distinguished from per-window predictive noise. Costs are separate observations. Descriptor behaviour is drawn from calibration low-volatility anchor cells, never hidden final outcomes. Numeric metadata are standardised on calibration; unavailable classical parameter counts are zero-imputed and do not establish actual zero parameter counts.',
        table(map_metrics),
        'The frozen independent map had a zero-centred prior for a positive loss scale. A separate calibration-only prior-sensitivity correction is recorded below. It changes no ACT acquisition result and is not silently substituted into the frozen map table.',table(prior_summary),
        '## 4. Dataset and chronological evaluation protocol',
        table(data[['ticker','rows','start','end','adequate']]),
        'Adjusted closes generate squared daily log returns; targets sum the next 1 or 5 trading-day squared returns. This is a daily-data volatility proxy, not intraday realised volatility. Origin-time descriptors use trailing data. Thresholds use training data only, with five origins removed at split edges. The training/calibration/test years are 2010–2016 / 2017–2018 / 2019–2020; 2010–2018 / 2019–2020 / 2021–2022; 2010–2020 / 2021–2022 / 2023–2025. The pool takes up to four evenly spaced origins per asset/cell, not every historical origin. Missing histories are not fabricated or forward-interpolated. Raw checksums match the download manifest.',
        '## 5. Candidate models and baselines',
        'Persistence, rolling mean, EWMA, HAR-style log-variance ridge, feature ridge, HistGB, Chronos-T5 Tiny and TimesFM 2.5 200M all completed all folds. Foundation inference ran on the RTX 4050; it was zero-shot, not GPU fine-tuning. Classical fits are deterministic; there is no stochastic-training-seed replication to report. TimesFM sign-flip augmentation is disabled in the frozen medium configuration. All outputs are positive variance forecasts with a 1e-10 floor. Baselines are fixed, random, risk-first, uncertainty-only, cost-aware optimistic/UCB-style and racing. The full-information oracle is evaluator-only and not deployable.',
        '## 6. Acquisition and stopping algorithms',
        'ACT estimates one-step Gaussian value of information: min(current posterior weighted means) minus expected min(updated posterior weighted means), divided by estimated action cost. Antithetic fantasy draws approximate this quantity. Batches contain 1 or 5 unseen windows; an uncertainty/cost fallback handles zero sampled improvement. The frozen rule requires joint posterior probability ≥0.95 of all twelve cell regrets ≤epsilon. It is stricter than twelve separate 95% marginal checks. Stop checks occur every ten actions and budget checkpoints.',
        'Epsilon equals clip(0.1 × median across cells of the cross-model SD of calibration mean normalised losses, 0.01, 0.1). The audit independently reproduces its value and raw/normalised equivalence. Numeric freeze manifests were materialised during simulation setup after reference forecasts were cached; the formula uses calibration only. This supplies functional leakage isolation, not a formal time-stamped preregistration before reference inference.',
        '## 7. Main results at matched budgets',
        'Mean ± SD across ten policy seeds, after averaging the three fixed folds within each seed. Original plots pooled 30 fold/seed rows for SD; the audit corrects this display without changing any mean or per-run regret.',table(main_table),
        'Racing has the lowest mean at 10% and 20%; ACT at 40%; UCB at 60%; ACT and risk-first tie at 80% to numerical precision. Cost caps match within each fold. Actual charged cost, origin counts, oracle agreement and worst-cell regret are retained for every checkpoint.',
        '![Audited regret](audit_v1/figures/decision_regret_vs_budget.png)',
        '## 8. Statistical analysis',
        'Policy seeds are acquisition replicates, not independent market histories. The audit resamples assets jointly across folds and calendar-quarter blocks within assets, preserving model pairing and overlapping-window clusters. It then resamples paired policy seeds. Folds and fitted models remain fixed; acquisition is not rerun inside bootstrap replicates. These are conditional intervals, not full deployment-level uncertainty.',
        f'The original bootstrap imputed unsupported asset/cells with pooled means. The versioned audit instead omits unsupported asset contributions and jointly excludes replicates without all target cells. It uses {bootstrap["used"]} replicates; the complete-support exclusion fraction is {bootstrap["exclusion_fraction"]:.2%}. This conditioning, eight asset clusters and only three periods remain limitations. Pointwise 95% intervals and six-comparison Bonferroni intervals are saved. Centred-bootstrap approximate p-values receive Holm correction over six baselines per budget and, additionally, all 30 comparisons. Prespecified paired Wilcoxon/Holm seed tests remain secondary.',
        table(comparisons[['budget_fraction','baseline','mean_difference','block_ci_low','block_ci_high','block_holm_p_six_baselines','block_holm_p_all_30']]),
        'Negative differences favour ACT. A numerically lower mean is not statistical superiority; isolated conditional comparisons do not establish superiority to adaptive baselines generally.',
        '## 9. Stopping-rule feasibility diagnosis',table(feasible_table),
        'For each model and cell, regret is L[m,c] − min_j L[j,c]. The best achievable single-model worst-cell regret is min_m max_c regret[m,c]. All twelve cells are supported and required; stress weights affect which global model is chosen but do not multiply a cell-specific tolerance. The global weighted oracle and minimax model need not coincide (notably fold 2). The same normalised loss and frozen epsilon are used in this calculation and in stopping.',
        'All three finite-reference minima exceed epsilon. Thus perfect knowledge of these empirical means would still fail the one-model/all-cell contract. Extra tests can reduce evidence uncertainty, but cannot change this finite-reference incompatibility. This is not proof that population expected losses have exactly these differences. The block analysis below quantifies sampling stability; no posterior uncertainty is substituted for actual empirical mean differences.',table(uncert[['fold','ci_low','ci_high','resampled_infeasible_fraction','price_ci_low','price_ci_high']]),
        'Model misspecification and calibration drift can add uncertainty, but are not needed to explain the observed point-estimate incompatibility. The Gaussian independent-cell-noise approximation also does not certify frequentist 95% commissioning confidence.',
        '![Feasibility](audit_v1/figures/full_information_regret_feasibility.png)',
        '## 10. Global versus regime-specific commissioning analysis',
        'This section is exploratory because the final periods had already been examined. Decision A can use the existing weighted global objective and the same calibration-derived epsilon on weighted regret. Its full-reference optimum has zero regret relative to the best single global candidate, so this scalar contract is feasible by construction. It relinquishes the original guarantee for every individual cell. Decision B allows one model per cell; its full-reference routing optimum has zero regret in every cell, also feasible by construction. These zero regrets refer to different action classes, not evidence of deployable perfect decisions.',
        'A calibration-only routing plan selects each cell’s calibration winner. Its final-period cell regrets are saved separately, as are calibration-only global-plan regrets. No alternative threshold or policy is tuned on final results. Posterior replay below uses only calibration and already revealed ACT observations, with separate diagnostic RNG seeds; it does not retrospectively change frozen recommendations. Feasibility with full information does not imply sufficient current evidence for joint regime-specific recommendations.',table(alternative),
        '![Alternative contracts](audit_v1/figures/exploratory_contract_coverage.png)',
        '## 11. Ablations and robustness',
        'All seven component removals and fixed-map selection completed. The no-uncertainty acquisition uses the predefined deterministic schedule fallback. No-abstention forces output; its frozen “confidently_wrong” field conflates unsafe forced recommendations with posterior confidence. The audit reports these separately; the primary abstention outcomes remain unchanged.',
        table(abst[abst.budget_fraction.eq(.8)][['policy','coverage','oracle_agreement_when_recommended','unsafe_all_cell_recommendation_rate','confidently_incorrect_rate']]),
        'Both six-asset commissioning pools completed with ten seeds and all three folds. Forecaster training and calibration remain the eight-asset fit; these are evaluation-pool sensitivity checks, not retrained asset-holdout experiments.',table(subplot),
        'Synthetic cases use an independent-cell Bayesian map to isolate acquisition behaviour. They are not financial observations and do not validate full descriptor generalisation. The 20% results include failures against simple policies:',table(syn),
        'Held-out-cell and leave-one-model-out calibration diagnostics are saved; the latter is exploratory with few model families. Model-selection probability scores use empirical calibration-holdout winners as noisy proxies, not independently known population winners.',
        '## 12. Online runtime and actual evaluation-cost analysis',table(online[['policy','wall_seconds','inference_seconds','preprocessing_seconds','acquisition_seconds','map_update_seconds','cache_hits','cache_misses','evaluation_calls','estimated_budget','actual_cost']]),
        f'Shared forecaster fitting/loading took {online.setup_shared_seconds.iloc[0]:.3f}s once and is not added three times. Mode B genuinely calls adapters on demand; cached reference losses are not used. No cached forecast is charged as a new inference. The run is one small pool, one seed and a fixed ACT/fixed/random order with shared loaded models, so timings do not establish a reproducible speed advantage. Preprocessing seconds cover per-origin slicing; initial data loading, calibration-map construction and artifact serialization are not fully included in the shared setup/wall intervals. Peak CUDA figures in the reference manifest are process high-water values, not isolated per-model memory measurements.',
        'Mode A charges exact saved per-window inference timings, with calibration maxima used to estimate action costs and evaluator-only timing quotes preventing cap overshoot. Acquisition and map-update overhead are recorded separately. Model inference dominates cost unevenly; the reference cost is heavily affected by the foundation candidates. Mode A measures simulated revelation quality and cannot show that precomputing the reference saved real compute.',
        '![Online cost](audit_v1/figures/online_execution_cost_breakdown.png)',
        '## 13. Limitations and threats to validity',
        'The small sampled pools, sparse asset/cell support, conditional block inference, shared training history, possible pretrained-data overlap with historical test years, approximate Gaussian loss/noise model, stochastic foundation sampling and a single online timing order limit claims. Foundation input chronology is checked, but independence of foundation pretraining from historical market periods cannot be certified. Raw/reference outcomes remain private to the evaluator by a cooperative in-process API; this is not an adversarial filesystem sandbox.',
        'Completed: all configured model/fold references, 420 primary policy trajectories, 2,100 primary budget rows, 180 subset trajectories, six synthetic conditions with ten seeds, map diagnostics, online execution, leakage/cost audits and stopping replay. Remaining outside this prototype: retraining or adaptive-policy replay within every block bootstrap replicate, independent future-period validation of alternative contracts, rigorous frequentist calibration of commissioning probabilities, and complete cold-start/resource profiling. These are not presented as completed experiments.',
        'Corrections are confined to versioned derived analyses: seed-level SD aggregation; confidence labels for forced outputs; support-aware bootstrap uncertainty; and a calibration-only independent-prior sensitivity comparison. Original forecasts, primary choices, metrics, thresholds and raw flags remain preserved. The corrected comparisons are a new audit analysis, not an independent confirmatory rerun.',
        '## 14. Reproducibility instructions',
        'Use the Windows virtual environment and pinned `requirements-lock.txt` described in README.md. The saved frozen config is `runs/frozen_69aaa8be2df3/results/manifests/config.json`. Reuse raw/reference caches; no inference is required for the audit.\n\n```powershell\n.\\.venv\\Scripts\\python.exe -m pytest -q\n.\\.venv\\Scripts\\python.exe scripts\\run_final_audit.py\n.\\.venv\\Scripts\\python.exe scripts\\generate_audited_report.py\n```\n',
        'Audit checkpoints, corrected tables, resampled truth draws and figures are under `results/audit_v1/`. Authoritative frozen policy rows remain `results/tables/policy_budget_results.csv`; refreshed statistical comparisons are explicitly labelled audit_v1, with originals preserved in the frozen snapshot. Dataset/checkpoint SHA256 manifests and source/run logs remain available. `next_evaluations.csv` contains posterior-only next-test suggestions; more evidence cannot itself resolve the finite-reference single-model trade-off.',
        '## 15. Final conclusions',
        conclusion,
        'A is not supported as a broad claim of superiority to the primary adaptive baselines. B is supported by lower numerical regret against fixed/random alongside uncertain adaptive comparisons. C should not be read as proof that ACT is uniformly inferior: racing/UCB win some budgets, while ACT wins others. D is supported for the finite reference and subject to the sampling uncertainty above; it does not establish irreducible population trade-offs. No epsilon was increased to manufacture recommendations.']
    report='\n\n'.join(lines)+'\n'
    chunks=re.split(r'(?m)^## \d+\. [^\n]+\n',report)
    body={i:chunks[i].strip() for i in range(1,16)}
    audit_tasks=pd.DataFrame([
        ['Reference coverage','completed','24 model/fold files; 8,400 final model-window losses','../../results/manifests/reference_validation.json'],
        ['Primary ledger integrity','passed','420 trajectories; 2,100 budget records; no duplicates/cost/loss mismatch','tables/ledger_audit.csv'],
        ['All-policy stopping replay','passed',f'2,100 evidence prefixes; {replay_manifest["decision_disagreements"]} decision disagreements','manifests/all_policy_replay.json'],
        ['Hidden-loss, budget, duplicate and stopping tests','passed','automated tests; see JUnit file after final verification','manifests/pytest.xml'],
        ['Frozen reporting checks','issues found; corrected separately','seed/fold SD, forced-confidence flag, missing-support bootstrap','tables/recommendation_flag_audit.csv'],
        ['Full-pool bootstrap stability','completed','4,000 supported paired asset/quarter replicates','tables/feasibility_block_uncertainty.csv'],
        ['Six-asset subset stability','completed','2,000 replicates per subset; all six fold/subset cases eligible','tables/subset_feasibility_stability.csv'],
        ['Global/routing contract diagnostics','completed; exploratory','same epsilon; no new inference or policy tuning','tables/alternative_contract_replay.csv'],
        ['Independent-map prior sensitivity','completed; calibration only','original zero-centred and new centred prior both retained','tables/corrected_map_prior_comparison.csv'],
        ['Population incompatibility certification','not established','non-smooth minimax, small clusters and broad simultaneous sensitivity','tables/simultaneous_feasibility_sensitivity.csv'],
        ['Independent prospective confirmation','incomplete; not launched','requires new preregistered holdout and aligned commissioning contract','report.md'],
        ['Full algorithm-retraining bootstrap / repeated runtime study','incomplete; not launched','current inference is conditional; online timing is one fixed-order check','report.md']
    ],columns=['task','status','evidence','artifact'])
    audit_tasks.to_csv(at/'audit_task_status.csv',index=False)
    claims=pd.DataFrame([
        ['Lower regret than fixed/random','Numerically yes at every budget; broad statistical superiority is not established','tables/policy_seed_summary.csv; tables/statistical_comparisons.csv'],
        ['Better than adaptive baselines','Mixed: racing wins low budgets; UCB wins 60%; ACT wins 40% and ties at 80%','tables/policy_seed_summary.csv'],
        ['Statistical support','No audited asset/time comparison survives six-baseline Holm at 5%; isolated pointwise intervals are insufficient','tables/statistical_comparisons.csv'],
        ['Actual runtime saving','Not demonstrated: ACT 7.305s, fixed 7.318s, random 7.130s; single fixed-order check','../../results/tables/online_runtime_results.csv'],
        ['Valid frozen recommendations','None issued. No single candidate meets all-cell epsilon in the empirical reference','tables/stopping_feasibility.csv; tables/abstention_summary.csv'],
        ['Primary bottleneck','Objective/contract mismatch plus candidate trade-offs in the finite reference; uncertainty also remains','tables/stopping_feasibility.csv; tables/alternative_contract_replay.csv'],
        ['Research judgement','Qualified B + finite-reference D; neither broad superiority nor population irreducibility is established','tables/statistical_comparisons.csv; tables/simultaneous_feasibility_sensitivity.csv']
    ],columns=['claim','assessment','saved_evidence'])
    claims.to_csv(at/'scientific_judgement.csv',index=False)
    evidence='\n\nEvidence: [ledger](tables/ledger_audit.csv), [all-policy replay](tables/all_policy_stopping_replay.csv), [loss-scale audit](tables/loss_scale_audit.csv), [statistics](tables/statistical_comparisons.csv), [task status](tables/audit_task_status.csv).'
    config_text='Frozen hash `'+cfg['hash']+'`. The assets, folds, model set, cell definitions, seeds, budgets, acquisition definitions, objective weights and calibrated tolerances are unchanged.\n\n```json\n'+json.dumps(cfg,indent=2)+'\n```\n\n'+body[2]+'\n\n'+body[6]
    variability=('Within each fold, SD is over ten acquisition seeds. Between folds, SD is descriptive dispersion of three fixed fold means, not an independent-market standard error. The aggregate first averages the three folds within each seed and then computes SD across ten paired seeds. No single pooled SD is used to imply independent financial replication. Market/window uncertainty is represented separately by paired block intervals.\n\n'
        +table(within[within.policy.eq('act')][['fold','budget_fraction','seed_mean','seed_std','n_policy_seeds']])+'\n\n'
        +table(fold_dispersion[fold_dispersion.policy.eq('act')][['budget_fraction','mean','fixed_fold_mean_sd','minimum_fold_mean','maximum_fold_mean']])
        +'\n\nSaved: [within-fold seed variability](tables/within_fold_seed_variability.csv), [between-fold variability](tables/between_fold_variability.csv), [fold-averaged seed variability](tables/policy_seed_summary.csv). Means and frozen choices do not change; uncertainty labels and SD values do.')
    stopping=body[9]+'\n\n### Asset-subset stability\n\n'+table(subset_feas[['subset','fold','epsilon','observed_minimum_worst_cell_regret','ci_low','ci_high','resampled_infeasible_fraction','minimum_cell_windows']])
    stopping+='\n\nThe all-cell requirement remains empirically infeasible in both subsets. All cells exceed the predefined six-window minimum. Subsets retain the same eight-asset calibration normalisation and epsilon. These are post hoc evaluator diagnostics, not new policy runs.\n\n### Limits of the stability calculation\n\nThe 2.5–97.5% resampling ranges describe stability of the empirical diagnostic; they are not certified population confidence bounds. Min-over-model/max-over-cell selection is nonsmooth and can be biased by sampling variability. A conservative simultaneous paired-difference sensitivity is therefore also shown:\n\n'+table(uniform)+'\n\nThis broader sensitivity does not establish population-level incompatibility. Evidence insufficiency and finite-reference model-choice incompatibility both remain relevant. Do not interpret a 100% resampled infeasibility frequency as posterior certainty about irreducible expected-loss differences. Saved: [full-pool stability](tables/feasibility_block_uncertainty.csv), [subset stability](tables/subset_feasibility_stability.csv), [simultaneous sensitivity](tables/simultaneous_feasibility_sensitivity.csv).'
    body[10]+='\n\nIn the ACT evidence replay, both exploratory contracts also issued zero confidence-qualified recommendations at every budget. The largest weighted-global near-optimality probability was '+f'{replay.exploratory_global_weighted_probability.max():.4f}'+', below 0.95; the largest routing joint probability was '+f'{replay.exploratory_routing_joint_probability.max():.4f}'+'. Thus changing the contract or action class alone did not establish evidence sufficiency under the current map. This replay retained the original acquisitions; it is not a test of a newly optimised routing policy. Source: [alternative-contract replay](tables/alternative_contract_replay.csv).'
    body[12]+='\n\nThe online check is on-demand execution on a fixed historical pool, not a live-market deployment study.'
    body[13]+='\n\nThe demonstrable specification bottleneck is the commissioning contract combined with the frozen candidate set. The data do not isolate whether the remaining posterior uncertainty comes primarily from map misspecification, limited window counts, noise or acquisition allocation. It would be unjustified to attribute all abstention to acquisition failure, or to claim acquisition success from provisional regret alone.'
    abstention='Every primary policy abstains at every budget. Thus conditional recommendation accuracy is undefined, and zero confidently wrong outputs is not evidence of successful commissioning. The forced no-abstention ablation is analysed separately. A recommendation is confidently wrong only when it is issued, its joint posterior confidence meets 0.95, and its reference worst-cell regret exceeds that fold’s frozen epsilon.\n\n'+table(abst[abst.budget_fraction.eq(.8)][['policy','coverage','oracle_agreement_when_recommended','unsafe_all_cell_recommendation_rate','confidently_incorrect_rate']])+'\n\nThe frozen flag labelled unsafe forced outputs as confident without checking confidence; the corrected flag changes that interpretation, not the original outputs or primary abstention results. See [flag audit](tables/recommendation_flag_audit.csv), [abstention summary](tables/abstention_summary.csv) and [figure](figures/abstention_rate_and_decision_quality.png).'
    replay_text='\n\nAll 420 trajectories and 2,100 budget states were reconstructed using only calibration and each policy’s revealed prefix. All saved posterior means match. Independent 1,024-draw Monte Carlo gate replay produces zero decision disagreements; original RNG states were not stored, so this is not a bitwise RNG replay. Forced-output exceptions are explicitly marked. [Replay manifest](manifests/all_policy_replay.json).'
    repro='\n\n### Reproduce the corrected audit without forecasting inference\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_final_audit.py\n.\\.venv\\Scripts\\python.exe scripts\\finish_audit_extensions.py\n.\\.venv\\Scripts\\python.exe scripts\\generate_audited_report.py\n.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=results/final_audit/manifests/pytest.xml\n.\\.venv\\Scripts\\python.exe scripts\\finalize_audit.py\n```\n\nThese commands read saved references and checkpoints. They do not download data, run adapters or change epsilon. Exact package versions are in `requirements-lock.txt`. Original CSVs/manifests remain unchanged in `results/` and the frozen snapshot; corrected outputs are exclusively in `results/final_audit/`, analysis `final_audit_v1`.'
    next_step='\n\n**Single most valuable next research step:** preregister one prospective commissioning study with a stopping contract aligned to the existing weighted-global objective, calibrate and check its near-optimality probabilities using calibration data only, and compare the unchanged ACT, UCB and racing policies on an untouched chronological holdout. This tests end-to-end commissioning and uncertainty sufficiency before adding another acquisition heuristic. No such extension or new forecasting run was launched here.'
    sections=[('Executive summary',body[1]+'\n\n'+table(claims)+'\n\n'+table(audit_tasks)),
        ('Frozen experimental configuration',config_text),('Data and model coverage',body[4]+'\n\n'+body[5]),
        ('Audit and leakage checks',body[3]+replay_text+evidence),('Main decision-regret results at matched budgets',body[7]),
        ('Statistical comparisons and uncertainty',body[8]),('Corrected seed and fold variability reporting',variability),
        ('Stopping-rule feasibility and bootstrap analysis',stopping),('Global versus regime-specific commissioning',body[10]),
        ('Abstention and confidently incorrect recommendations',abstention),('Online runtime versus simulated evaluation cost',body[12]),
        ('Ablations and robustness',body[11]+'\n\n[All ablation checkpoint outcomes](../../results/tables/ablation_results.csv); [six-asset policy results](../../results/tables/asset_subset_robustness.csv); [synthetic results](../../results/tables/synthetic_robustness_results.csv).'),
        ('Limitations and unresolved issues',body[13]),('Final conclusion and recommended next research step',body[15]+next_step+repro)]
    report='# ACT-TSFM — Final Scientific Audit\n\nAnalysis `final_audit_v1`; frozen experiment `69aaa8be2df3`. Corrected and exploratory derivatives only; no original CSV or manifest is replaced.\n\n'+'\n\n'.join(f'## {i}. {title}\n\n{text}' for i,(title,text) in enumerate(sections,1))+'\n'
    report=report.replace('audit_v1/figures/','figures/')
    (FINAL/'report.md').write_text(report,encoding='utf-8')
    summary=conclusion+'\n\n'+table(main_table)+'\n\n'+table(feasible_table)+'\n\nOnline check:\n'+table(online[['policy','wall_seconds','evaluation_calls']])+'\n'
    (FINAL/'summary.md').write_text(summary,encoding='utf-8')
    save_json(FINAL/'manifests/report_artifacts.json',dict(analysis_id='final_audit_v1',frozen_hash=cfg['hash'],
        report='results/final_audit/report.md',figures=[str(p.relative_to(ROOT)) for p in files],
        derived_root_replacements=[],
        primary_observations_modified=False,conclusion=conclusion))

