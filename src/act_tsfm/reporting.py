import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT
from .statistical_tests import comparisons

def report(cfg):
    tab=ROOT/'results/tables';fig=ROOT/'results/figures'
    r=pd.read_csv(tab/'policy_budget_results.csv');comparison=comparisons(r)
    comparison.to_csv(tab/'statistical_comparisons.csv',index=False)
    from .block_bootstrap import compare
    blocked=compare(cfg,r)
    blocked.to_csv(tab/'block_bootstrap_comparisons.csv',index=False)
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def plot(d,x,y,title,file):
        f,ax=plt.subplots(figsize=(8,4.5))
        for name,g in d.groupby('policy'):
            if name not in ['act','fixed','random','risk','ucb','racing','uncertainty']:continue
            if x=='cost':
                s=g.groupby('budget_fraction').agg(cost_mean=('cost','mean'),y_mean=(y,'mean'),y_std=(y,'std')).fillna(0)
                ax.errorbar(s.cost_mean,s.y_mean,yerr=s.y_std,marker='o',capsize=2,label=name)
                continue
            s=g.groupby(x)[y].agg(['mean','std']).fillna(0)
            ax.plot(s.index,s['mean'],marker='o',label=name)
            ax.fill_between(s.index,s['mean']-s['std'],s['mean']+s['std'],alpha=.08)
        ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '),title=title)
        if y=='accuracy_among_recommended' and d[d.policy.isin(['act','fixed','random','risk','ucb','racing','uncertainty'])][y].isna().all():
            ax.text(.5,.5,'All primary policies abstained;\nconditional accuracy is undefined',transform=ax.transAxes,ha='center')
        ax.legend(ncol=4,fontsize=8);f.tight_layout();f.savefig(fig/file);plt.close(f)
    for metric,title,file in [('regret','Decision regret (mean ± policy-seed SD)','decision_regret_vs_budget.png'),
        ('agreement','Agreement with finite reference optimum','oracle_agreement_vs_budget.png'),
        ('worst_regime_regret','Worst cell regret','worst_regime_regret_vs_budget.png')]:
        plot(r,'budget_fraction',metric,title,file)
    plot(r,'cost','regret','Regret versus calibrated inference charge','regret_vs_compute_cost.png')
    pool=pd.read_csv(ROOT/f'artifacts/oracle_reference/{cfg["hash"]}/fold1/test_windows.csv')
    occ=pool.groupby(['cell','asset']).size().unstack(fill_value=0).reindex(range(12),fill_value=0)
    f,ax=plt.subplots(figsize=(9,4));occ.plot.bar(stacked=True,ax=ax)
    ax.set(xlabel='Cell (volatility × trend × horizon)',ylabel='Windows',title='Balanced evaluation-pool occupancy')
    f.tight_layout();f.savefig(fig/'regime_cell_occupancy.png');plt.close(f)
    perf=pd.read_csv(tab/'model_forecasting_performance.csv')
    matrix=perf.groupby(['model','cell']).loss.mean().unstack()
    f,ax=plt.subplots(figsize=(9,4));im=ax.imshow(matrix,aspect='auto');ax.set_yticks(range(len(matrix)),matrix.index)
    ax.set(xlabel='Cell (even: h=1, odd: h=5)',title='Mean normalised absolute log variance error')
    f.colorbar(im,ax=ax,label='Loss');f.tight_layout();f.savefig(fig/'model_performance_by_regime.png');plt.close(f)
    mm=pd.read_csv(tab/'performance_map_metrics.csv')
    f,ax=plt.subplots(figsize=(8,4));mm.groupby('map')[['coverage95','width95']].mean().plot.bar(ax=ax)
    ax.set(title='Chronological calibration holdout: intervals',ylabel='Coverage / interval width')
    f.tight_layout();f.savefig(fig/'posterior_interval_calibration.png');plt.close(f)
    predictions=pd.concat([pd.read_csv(p) for p in tab.glob('map_predictions_fold*.csv')])
    f,ax=plt.subplots(figsize=(5,5));ax.scatter(predictions.predicted,predictions.observed,s=8,alpha=.3)
    lim=max(predictions.predicted.max(),predictions.observed.max());ax.plot([0,lim],[0,lim],'k--')
    ax.set(xlabel='Posterior mean loss',ylabel='Observed window loss',title='Held-out calibration predictions')
    f.tight_layout();f.savefig(fig/'predicted_vs_observed_loss.png');plt.close(f)
    f,ax=plt.subplots(figsize=(6,4));ax.hist((predictions.observed-predictions.predicted)/predictions.sd,bins=40)
    ax.set(xlabel='Standardised predictive residual',ylabel='Windows',title='Posterior predictive uncertainty check')
    f.tight_layout();f.savefig(fig/'posterior_uncertainty.png');plt.close(f)
    f,ax=plt.subplots(figsize=(9,4));r.groupby('policy').regret.mean().sort_values().plot.bar(ax=ax)
    ax.set(ylabel='Mean decision regret',title='Policies and component ablations')
    f.tight_layout();f.savefig(fig/'ablation_comparisons.png');plt.close(f)
    f,ax=plt.subplots(figsize=(8,4));r.groupby('policy')[['acquisition_seconds','map_update_seconds','cost']].mean().plot.bar(stacked=True,ax=ax)
    ax.set(ylabel='Seconds',title='Simulation overhead and calibrated inference charge')
    f.tight_layout();f.savefig(fig/'runtime_breakdown.png');plt.close(f)
    abst=pd.read_csv(tab/'abstention_summary.csv')
    plot(abst,'recommendation_fraction','accuracy_among_recommended','Coverage versus accuracy among recommendations','abstention_quality.png')
    traces=list((ROOT/f'artifacts/observed_losses/{cfg["hash"]}').glob('fold1_act_42.json'))
    if traces:
        trace=pd.DataFrame(json.loads(traces[0].read_text())['trace'])
        f,ax=plt.subplots(figsize=(8,4));ax.scatter(trace.step,trace.pair,c=trace.batch,s=14,cmap='viridis')
        ax.set(xlabel='Decision step',ylabel='Model × cell index',title='ACT selected interactions (seed 42)')
        f.tight_layout();f.savefig(fig/'acquisition_selected_interactions.png');plt.close(f)
    summary=r[r.policy.isin(['fixed','random','risk','uncertainty','ucb','racing','act'])].groupby(['budget_fraction','policy']).regret.agg(['mean','std']).reset_index()
    summary.to_csv(tab/'summary.csv',index=False)
    best=summary.loc[summary.groupby('budget_fraction')['mean'].idxmin()]
    status_path=ROOT/f'results/manifests/model_status_{cfg["hash"]}.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else perf.groupby(['fold','model']).size().reset_index(name='cell_records').assign(status='forecast CSVs complete').to_dict('records')
    lines=['# ACT-TSFM — executed research prototype',
      '\n## Abstract\nA conjugate Bayesian interaction map and cost-aware one-step value-of-information policy were implemented. Results below come from saved ETF forecasts and sequential revelation logs. Scope: '+cfg.get('scope','medium')+'.',
      '\n## Problem and method\nModels are selected by prespecified weighted cell loss. Twelve cells combine training-only volatility tertiles, past 20-day trend sign and horizons 1/5. Loss is absolute log(predicted variance / observed variance), divided by calibration median. The Bayesian map uses shrinkage Gaussian coefficients, static and anchor behavioural descriptors, bilinear interactions and model/cell effects. Calibration cell noise is partially pooled. A fixed 0.1 discount on calibration counts allows some temporal change; it is an approximation, not a learned dynamic model.',
      '\n## Acquisition and stopping\nGaussian fantasy observations update weighted posterior model means. Reduction in expected minimum posterior loss estimates one-step Bayes regret reduction. It is divided by batch inference charge. Batches are 1 or 5. When sampled improvement is zero, a frozen uncertainty/cost fallback runs. A joint posterior probability of at least .95 that the chosen model is epsilon-optimal in every cell permits recommendation; epsilon is computed only from calibration. Otherwise the system records insufficient evidence and lacking cells.',
      '\n## Data and chronology\nAdjusted Yahoo daily closes produce squared log returns and sums of future 1/5-day squares. These are daily-data volatility proxies, not intraday realised volatility. Features use origin-time information; five origins are removed from each split edge. No missing values are interpolated. Pools use deterministic equal per-asset/cell sample caps. Histories, targets, thresholds, occupancy and checksums are saved separately. Foundation pretraining datasets may overlap historical evaluation years; zero-shot chronology cannot certify pretraining independence.',
      '\n## Actual run status\n```json\n'+json.dumps(status,indent=2)+'\n```',
      '\n## Primary endpoint: decision regret at matched calibrated inference charge\n'+best.to_string(index=False),
      '\n## Paired comparisons\nACT minus baseline regret; negative favours ACT. Intervals resample policy seeds, conditional on these fixed folds.\n```\n'+comparison.to_string(index=False)+'\n```',
      '\n## Baselines and ablations\nFixed, random, stress-first, uncertainty, cost-aware uncertainty/UCB-style and racing policies share calibration evidence and a reveal-only API. The oracle is the finite-pool minimum computed on the evaluator side. All requested component removals are run, but no-uncertainty uses a deterministic schedule rather than a sophisticated deterministic decision policy. The low-rank map uses a two-dimensional calibration SVD embedding. The UCB-style implementation is an information/cost heuristic, not a formally derived regret-bound algorithm.',
      '\n## Budget and computational limitations\nMode A is simulated revelation of a precomputed matrix. Actions use conservative calibration maxima per model/cell as estimated costs. Each revealed observation is charged its actual saved inference timing. Evaluator-only timing quotes prevent actual budget overrun; quotes expose no loss. The finite pool cost defines each matched budget. Acquisition and map-update overhead are measured separately. Stop checks occur every ten actions and at budget checkpoints. Mode A demonstrates decision quality under simulated revelation; it does not establish actual compute savings.',
      '\n## Statistical uncertainty and incomplete work\nPolicy seeds 42–51 are used. Paired policy-seed bootstrap intervals and Wilcoxon/Holm secondary tests are available. Paired asset/quarter-block intervals appear below. Held-out-cell and leave-one-model-out calibration diagnostics are saved in generalisation_metrics.csv; the latter is exploratory with only eight candidates. Synthetic robustness includes equal/stable/changing rankings, noisy stress, sparse cells and sharp cost variation. Asset-subset policy reruns and explicit calibration of model-selection probabilities remain incomplete. Treat comparisons as exploratory. Sparse cells halt a financial fold under the predefined rule. The six classical adapters are deterministic; HistGB has no stochastic sampling. Foundation inference is zero-shot, without repeated training seeds. Chronos uses seeded sampling. Initial calibration evaluation cost is shared evidence and is not charged to simulated final-period budgets; include this amortised cost when assessing deployment economics.',
      '\n## Conclusions\nThe saved results establish execution of the listed prototype scope only. They do not establish a publishable superiority or practical compute-saving claim. Review ablation and abstention tables alongside regret: low regret from an abstaining method is not evidence of successful commissioning.',
      '\n## Reproduction\nSee README.md and results/manifests/config.json. Reuse the identical configuration hash for cached forecasts.']
    (ROOT/'results/report.md').write_text('\n'.join(lines),encoding='utf-8')
    online=tab/'online_runtime_results.csv'
    if online.exists():
        with (ROOT/'results/report.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\n## Mode B: actual online execution\nShared model setup is reported separately. Independent policy caches; selected forecasts actually run on demand. Regret was not estimated in this runtime-only check. Costs can exceed the estimated cap by the final on-demand action.\n```\n'+pd.read_csv(online).to_string(index=False)+'\n```\n')
    with (ROOT/'results/report.md').open('a',encoding='utf-8') as stream:
        stream.write('\n\n## Paired asset/time-block uncertainty\nCalendar quarters form time blocks; assets are resampled jointly across folds, and policy seeds are paired. This resamples evaluator losses conditional on recorded decisions. It does not account fully for model-training dependence or rerun adaptive selection. Sparse resampled cells fall back to the original pooled cell mean, so the intervals are exploratory.\n```\n'+blocked.to_string(index=False)+'\n```\n')
        stream.write('\n\n## Performance-map results\nChronological calibration holdout, averaged over executed folds. The low-rank embedding and all descriptors are constructed without final-test outcomes.\n```\n'+mm.groupby('map')[['rmse','mae','coverage95','width95']].mean().to_string()+'\n```\n')
        stream.write('\n\n## Ablation results\nAll policies are assessed using the same stress-weighted primary endpoint; equal-weight regret is a secondary saved column.\n```\n'+r.groupby(['budget_fraction','policy']).regret.mean().unstack().to_string()+'\n```\n')
        stream.write('\n\n## Abstention results\nA missing conditional accuracy means that no recommendation was made. Provisional model-choice regret is reported even when the method abstains.\n```\n'+abst.to_string(index=False)+'\n```\n')
        synthetic=tab/'synthetic_robustness_results.csv'
        if synthetic.exists():
            sr=pd.read_csv(synthetic)
            stream.write('\n\n## Synthetic stress and failure cases\nThese are separately labelled synthetic experiments, not financial observations.\n```\n'+sr[sr.budget_fraction.eq(.2)].groupby(['scenario','policy']).regret.mean().unstack().to_string()+'\n```\n')
        means=r.groupby(['budget_fraction','policy']).regret.mean().unstack()
        coverage=float(abst[abst.policy.eq('act')].recommendation_fraction.mean())
        stream.write(f'\n\n## Concise empirical summary\nACT mean provisional regret was below fixed selection at {int((means.act<means.fixed).sum())}/{len(means)} budgets and below random selection at {int((means.act<means.random).sum())}/{len(means)} budgets. Its recommendation fraction averaged {coverage:.1%} over budget checkpoints. These descriptive counts are not significance claims; use the paired block intervals above.\n')
    (ROOT/'results/summary.txt').write_text(best.to_string(index=False)+'\n\n'+comparison.to_string(index=False),encoding='utf-8')
