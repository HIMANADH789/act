import copy
import numpy as np
import pandas as pd
import pytest
from act_tsfm.commissioning_diagnostics import regret_contract,posterior_contracts,corrected_recommendation_flags
from act_tsfm.stopping import decision
from act_tsfm.regime_cells import weights
from act_tsfm.performance_map import BayesianMap
from act_tsfm.evaluation import RestrictedEvaluator
from act_tsfm.simulator import run,POLICIES,ABLATIONS
from act_tsfm.config import load

def test_global_and_routing_have_different_feasible_contracts():
    loss=np.array([[0.,.3]*6,[.3,0.]*6])
    out=regret_contract(loss,weights(False),.04)
    assert out['primary_weighted_regret']==0
    assert out['minimum_worst_cell_regret']==pytest.approx(.3)
    assert not out['all_cell_single_model_feasible']
    assert out['global_weighted_contract_feasible']
    assert out['regime_specific_contract_feasible']
    assert out['price_of_single_model']==pytest.approx(.15)
    assert not decision(np.tile(loss.ravel(),(100,1)),weights(False),.04)['recommend']

def test_loss_and_tolerance_normalisation_is_consistent():
    loss=np.array([[.1,.7]*6,[.5,.2]*6]);scale=2.7
    a=regret_contract(loss,weights(),.1);b=regret_contract(loss/scale,weights(),.1/scale)
    assert a['all_cell_single_model_feasible']==b['all_cell_single_model_feasible']
    assert a['minimum_worst_cell_regret']/scale==pytest.approx(b['minimum_worst_cell_regret'])
    with pytest.raises(ValueError):regret_contract(np.full((2,12),np.nan),weights(),.1)

def test_evidence_insufficiency_differs_from_tradeoffs():
    # Expected means tie, so an epsilon-optimal global model exists, but evidence is ambiguous.
    d=np.ones((100,2,12));d[:50,0]=0;d[50:,1]=0
    p=posterior_contracts(d,0,weights(),.01)
    assert p['primary_selected_joint_probability']==.5
    assert regret_contract(d.mean(0),weights(),.01)['all_cell_single_model_feasible']
    assert not decision(d.reshape(100,-1),weights(),.01)['recommend']

def test_stopping_is_joint_not_twelve_marginal_tests():
    d=np.full((100,2,12),.05);d[:,0,:]=0
    for c in range(12):d[c*4:c*4+4,0,c]=.1
    regret=d[:,0,:]-d.min(1)
    assert np.all((regret<=.01).mean(0)==.96)
    verdict=decision(d.reshape(100,-1),weights(),.01)
    assert verdict['probability']==pytest.approx(.52)
    assert not verdict['recommend']

def test_forced_recommendation_is_not_automatically_confident():
    assert corrected_recommendation_flags(True,.1,.3,.04)==dict(incorrect_recommendation=1,confidently_incorrect=0)
    assert corrected_recommendation_flags(True,.99,.3,.04)['confidently_incorrect']==1

def test_end_to_end_hidden_loss_invariance_all_policies():
    cfg=load();cfg.update(budgets=[.05],posterior_draws=16,fantasies=4,_full_cost=72.)
    records=pd.DataFrame([dict(pair=p,loss=1.,cost=1.,key=f'{p}/{j}') for p in range(24) for j in range(3)])
    truth=np.ones((2,12));initial=BayesianMap(np.eye(24),np.ones(24))
    for name in POLICIES+ABLATIONS:
        ev=RestrictedEvaluator(records,24,42)
        _,before=run(name,ev,initial,np.ones(24),truth,weights(),cfg,42,.01)
        seen={r['key'] for r in ev.log}
        changed=records.copy();ix=next(i for i,k in enumerate(changed.key) if k not in seen)
        changed.loc[ix,'loss']=1e9
        altered=RestrictedEvaluator(changed,24,42)
        _,after=run(name,altered,initial,np.ones(24),truth*999,weights(),cfg,42,.01)
        assert before==after
