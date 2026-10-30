"""Small genuinely on-demand execution check, independent per-policy observation caches."""
import time
import numpy as np
import pandas as pd
from .config import ROOT,save_json
from .preprocessing import process,split,FEATURES
from .regime_cells import weights
from .forecasting_adapters import Adapter,sync
from .runner import setup_map
from .simulator import run
from .evaluation import RestrictedEvaluator

class OnlineEvaluator(RestrictedEvaluator):
    def __init__(self,pool,adapters,data,cost,seed,fold):
        super().__init__(pool,len(cost),seed)
        self.adapters,self.data,self.cost,self.fold=adapters,data,cost,fold
        self.cache={};self.hits=0;self.misses=0;self.inference=0.;self.preprocessing=0.

    def reveal(self,action):
        metadata=super().reveal(action)
        out=[]
        for row in metadata:
            key=row['key']
            if key in self.cache:
                self.hits+=1;out.append(self.cache[key]);continue
            self.misses+=1
            tick=time.perf_counter()
            history=self.data[row['asset']].loc[:pd.Timestamp(row['date']),FEATURES]
            self.preprocessing+=time.perf_counter()-tick
            sync();tick=time.perf_counter()
            pred=self.adapters[row['model']].predict(history,int(row['h']))
            sync();elapsed=time.perf_counter()-tick;self.inference+=elapsed
            result=dict(row,prediction=pred,loss=abs(np.log(max(row['target'],1e-10)/pred))/row['scale'],cost=elapsed)
            self.cache[key]=result;out.append(result)
        # Base class logged metadata; replace with actual observations.
        self.log[-len(out):]=out
        return out

def online(cfg):
    data=process(cfg);folder=ROOT/f'artifacts/oracle_reference/{cfg["hash"]}/fold1'
    names=[n for n in ['persistence','rolling','ewma','har','ridge','histgb','chronos','timesfm'] if (folder/f'{n}_cal.csv').exists()]
    cal=pd.concat([pd.read_csv(folder/f'{n}_cal.csv') for n in names]);scale=max(cal.loss.median(),.1);cal.loss/=scale
    cal['pair']=cal.model.map({n:i for i,n in enumerate(names)})*12+cal.cell
    cost=cal.groupby('pair').cost.max().reindex(range(len(names)*12)).to_numpy()
    initial,_=setup_map(cal,names,cost,pd.read_csv(folder/'cal_windows.csv'))
    train={t:split(d,2010,cfg['folds'][0][0]) for t,d in data.items()}
    tick=time.perf_counter();adapters={n:Adapter(n,cfg['device'],cfg['context'],flip_invariance=cfg.get('timesfm_flip_invariance',False)).fit(train) for n in names}
    setup_seconds=time.perf_counter()-tick
    wp=pd.read_csv(folder/'test_windows.csv').groupby(['asset','cell']).head(1)
    pool=pd.concat([wp.assign(model=n,pair=i*12+wp.cell) for i,n in enumerate(names)],ignore_index=True)
    pool['key']=pool.model+'/'+pool.asset+'/'+pool.date+'/'+pool.h.astype(str)
    pool['cost']=pool.pair.map(dict(enumerate(cost)));pool['scale']=scale
    rows=[]
    # No oracle outcomes are loaded. Regret is unavailable in this online-only check.
    for policy in ['act','fixed','random']:
        ev=OnlineEvaluator(pool,adapters,data,cost,42,1)
        tick=time.perf_counter()
        cc=dict(cfg,budgets=[.2],_full_cost=pool.cost.sum())
        results,trace=run(policy,ev,initial,cost,np.zeros((len(names),12)),weights(),cc,42,.01)
        r=results[-1]
        rows.append(dict(policy=policy,wall_seconds=time.perf_counter()-tick,setup_shared_seconds=setup_seconds,
            inference_seconds=ev.inference,preprocessing_seconds=ev.preprocessing,acquisition_seconds=r['acquisition_seconds'],
            map_update_seconds=r['map_update_seconds'],cache_hits=ev.hits,cache_misses=ev.misses,evaluation_calls=len(ev.log),
            estimated_budget=.2*pool.cost.sum(),actual_cost=r['cost'],models=','.join(names)))
        save_json(ROOT/f'artifacts/observed_losses/{cfg["hash"]}/online_{policy}.json',dict(trace=trace,observations=ev.log))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/online_runtime_results.csv',index=False)
