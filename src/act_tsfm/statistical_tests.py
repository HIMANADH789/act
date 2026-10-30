"""Secondary paired seed analysis. Serial block-bootstrap is explicitly pending."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows=[];rng=np.random.default_rng(42)
    for budget,g in results.groupby('budget_fraction'):
        pivot=g.groupby(['seed','policy']).regret.mean().unstack()
        for baseline in ['fixed','random','risk','uncertainty','ucb','racing']:
            d=(pivot['act']-pivot[baseline]).dropna().to_numpy()
            boot=rng.choice(d,(2000,len(d)),replace=True).mean(1)
            p=wilcoxon(d).pvalue if np.any(d) else 1.
            rows.append(dict(budget_fraction=budget,baseline=baseline,mean_difference=d.mean(),
                ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),p_value=p,
                unit='paired policy seeds, folds averaged; conditional on fixed evaluation periods',
                limitation='not asset/time block-bootstrap; no deployment-level inference'))
    out=pd.DataFrame(rows)
    out['holm_p']=1.
    for _,g in out.groupby('budget_fraction'):
        ix=g.sort_values('p_value').index
        out.loc[ix,'holm_p']=np.minimum(1,np.maximum.accumulate(out.loc[ix,'p_value'].to_numpy()*np.arange(len(ix),0,-1)))
    return out
