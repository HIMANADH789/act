import numpy as np
import pandas as pd
import pytest
from act_tsfm.preprocessing import features,FEATURES,split
from act_tsfm.performance_map import BayesianMap,design_matrix
from act_tsfm.evaluation import RestrictedEvaluator,Action
from act_tsfm.acquisition import choose
from act_tsfm.stopping import decision
from act_tsfm.regime_cells import cell_id,weights

def raw():
    rng=np.random.default_rng(42)
    return pd.DataFrame({'Adj Close':100*np.exp(np.cumsum(rng.normal(0,.01,400))),
                         'Close':100.,'Volume':1000},index=pd.bdate_range('2016-01-01',periods=400))

def test_targets_and_past_only():
    r=raw();a=features(r);changed=r.copy();changed.iloc[250:,0]*=2;b=features(changed)
    pd.testing.assert_frame_equal(a[FEATURES].iloc[:250],b[FEATURES].iloc[:250])
    assert a.rv5.iloc[100]==pytest.approx(a.v.iloc[101:106].sum())
    assert a.rv1.iloc[100]==pytest.approx(a.v.iloc[101])
    assert a.outcome5.iloc[100]==a.index[105]
    assert not any('rv' in x or 'outcome' in x for x in FEATURES)
    q=split(a,2016,2016)
    assert q.outcome5.max().year==2016

def test_cells():
    assert {cell_id(pd.Series({'vol':v,'trend':t}),h,np.array([.1,.2]))
            for v in [.05,.15,.3] for t in [-1,1] for h in [1,5]}==set(range(12))

def test_gaussian_posterior_matches_closed_form():
    x=np.array([[1.,0.],[1.,1.],[1.,2.]])
    y=np.array([1.,2.,3.]);p=BayesianMap(x,np.ones(3),ridge=2)
    for j in range(3):p.update(j,y[j])
    cov=np.linalg.inv(2*np.eye(2)+x.T@x)
    np.testing.assert_allclose(p.cov,cov)
    np.testing.assert_allclose(p.mean,cov@x.T@y)
    assert np.linalg.eigvalsh(p.predict()[1]).min()>-1e-10

def test_duplicate_and_hidden_outcome_invariance():
    r=pd.DataFrame([dict(pair=i,loss=float(i),cost=1,key=str(i)) for i in range(24)])
    e=RestrictedEvaluator(r,24,42);r.loc[3,'loss']=1e9;f=RestrictedEvaluator(r,24,42)
    args=dict(kind='act',mu=np.zeros(24),cov=np.eye(24),noise=np.ones(24),cost=np.ones(24),
              remaining=10,w=weights(),step=0)
    a=choose(**args,available=e.counts(),rng=np.random.default_rng(42))
    b=choose(**args,available=f.counts(),rng=np.random.default_rng(42))
    assert a==b
    e.reveal(a)
    with pytest.raises(ValueError): e.reveal(a)
    assert not hasattr(e,'records')

def test_equal_losses_no_superiority_claim():
    verdict=decision(np.ones((100,24)),weights(),.01)
    assert verdict['recommend']  # Any model is epsilon-optimal; not unique superiority.
    assert verdict['probability']==1

def test_regime_switch_abstains():
    d=np.array([0.,1.]*6+[1.,0.]*6)
    verdict=decision(np.tile(d,(100,1)),weights(),.1)
    assert not verdict['recommend']

def test_descriptor_permutation_and_ablation():
    z=np.array([[0.,1.],[1.,0.]])
    x=np.arange(12)[:,None]
    a=design_matrix(z,x);b=design_matrix(z[::-1],x)
    assert not np.allclose(a,b)
    assert design_matrix(z,x,'no_effects').shape[1]<a.shape[1]

def test_every_policy_cost_budget_and_revelation():
    from act_tsfm.simulator import run,POLICIES,ABLATIONS
    from act_tsfm.config import load
    cfg=load();cfg.update(budgets=[.1,.2],posterior_draws=16,fantasies=4,_full_cost=72.)
    truth=np.ones((2,12));records=pd.DataFrame([dict(pair=p,loss=1.,cost=1.,key=f'{p}/{j}') for p in range(24) for j in range(3)])
    for name in POLICIES+ABLATIONS:
        ev=RestrictedEvaluator(records,24,42)
        p=BayesianMap(np.eye(24),np.ones(24))
        results,trace=run(name,ev,p,np.ones(24),truth,weights(),cfg,42,.01)
        assert all(r['cost']<=r['budget_fraction']*72+1e-8 for r in results)
        keys=[q['key'] for q in ev.log]
        assert len(set(keys))==len(keys)
        assert all(r['regret']==0 for r in results)
