import copy
import time
import numpy as np
from .acquisition import choose
from .evaluation import Action
from .stopping import decision
from .regime_cells import weights

POLICIES=['fixed','random','risk','uncertainty','ucb','racing','act']
ABLATIONS=['no_decision','no_cost','no_descriptors','no_behaviour','no_uncertainty','no_weight','no_abstention']

def run(policy, evaluator, initial, cost, truth, w, cfg, seed, epsilon):
    rng=np.random.default_rng(seed)
    posterior=copy.deepcopy(initial)
    full_cost=cfg['_full_cost']
    spent=0.; step=0; acquisition=0.; updates=0.; results=[]; trace=[]
    stopped=False
    for fraction in cfg['budgets']:
        budget=fraction*full_cost
        while not stopped:
            mu,cov=posterior.predict()
            tick=time.perf_counter()
            a=choose(policy,mu,cov,posterior.noise,cost,evaluator.counts(),budget-spent,w,rng,step,cfg['fantasies'],cfg['batch_sizes'])
            acquisition+=time.perf_counter()-tick
            if a is None: break
            if evaluator.quote(a)>budget-spent:
                affordable=[Action(p,1) for p,n in enumerate(evaluator.counts()) if n and evaluator.quote(Action(p,1))<=budget-spent]
                if not affordable: break
                a=min(affordable,key=lambda v:cost[v.pair])
            out=evaluator.reveal(a)
            # Never exceed actual measured budget, estimated costs are conservative maxima.
            spent+=sum(r['cost'] for r in out)
            tick=time.perf_counter()
            posterior.update(a.pair,float(np.mean([r['loss'] for r in out])),len(out))
            updates+=time.perf_counter()-tick
            step+=1
            verdict=decision(posterior.draws(rng,cfg['posterior_draws']),w,epsilon,cfg['confidence']) if step%10==0 else dict(selected=int((mu.reshape(-1,12)@w).argmin()),recommend=False,probability=0)
            trace.append(dict(step=step,pair=a.pair,batch=a.size,cost=spent,selected=verdict['selected'],
                probability=verdict['probability'],observations=[r['key'] for r in out],posterior_mean=posterior.predict()[0].tolist()))
            if verdict['recommend'] and policy!='no_abstention': stopped=True
        verdict=decision(posterior.draws(rng,cfg['posterior_draws']),w,epsilon,cfg['confidence'])
        chosen=verdict['selected']; primary_w=weights();oracle=int((truth@primary_w).argmin())
        regime=truth[chosen]-truth.min(0)
        recommend=verdict['recommend'] or policy=='no_abstention'
        results.append(dict(policy=policy,seed=seed,budget_fraction=fraction,cost=spent,evaluations=len(evaluator.log),
            regret=float(truth[chosen]@primary_w-(truth@primary_w).min()),equal_weight_regret=float(truth[chosen].mean()-truth.mean(1).min()),worst_regime_regret=float(regime.max()),
            agreement=int(chosen==oracle),selected=chosen,oracle=oracle,abstain=int(not recommend),
            confidently_wrong=int(recommend and regime.max()>epsilon),probability=verdict['probability'],
            lacking_cells=str(verdict['lacking_cells']),acquisition_seconds=acquisition,map_update_seconds=updates,
            **{f'cell_{c}_regret':regime[c] for c in range(12)}))
    return results,trace
