"""Pipeline CPU policy work as each GPU reference fold becomes ready.

Run after starting `act-tsfm reference`. Reads the frozen active run manifest once.
No policy receives incomplete final matrices; this script is evaluator orchestration.
"""
import copy
import json
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
from act_tsfm.config import ROOT,save_json
from act_tsfm.forecasting_adapters import NAMES
from act_tsfm.runner import simulations
from act_tsfm.online_runner import online
from act_tsfm.robustness import synthetic
from act_tsfm.reporting import report

cfg=json.loads((ROOT/'results/manifests/config.json').read_text())
status_path=ROOT/f'results/manifests/pipeline_{cfg["hash"]}.json'
record=dict(hash=cfg['hash'],status='running',start=datetime.now(timezone.utc).isoformat())
save_json(status_path,record)
try:
    for fold in range(1,len(cfg['folds'])+1):
        folder=ROOT/f'artifacts/oracle_reference/{cfg["hash"]}/fold{fold}'
        deadline=time.monotonic()+3600
        while True:
            names=[n for n in NAMES if cfg['foundation_models'] or n not in ('chronos','timesfm')]
            if all((folder/f'{n}_test.csv').exists() for n in names):break
            statuses=ROOT/f'results/manifests/model_status_{cfg["hash"]}.json'
            if statuses.exists():
                s=[r for r in json.loads(statuses.read_text()) if r['fold']==fold]
                if len(s)==len(names) and all(r['status'] in ('complete','cached','failed') for r in s):break
            if time.monotonic()>deadline:raise TimeoutError(f'Reference fold {fold} not ready')
            time.sleep(10)
        print('SIMULATE_READY_FOLDS',fold,flush=True)
        partial=copy.deepcopy(cfg);partial['folds']=cfg['folds'][:fold]
        simulations(partial)
        record['completed_policy_folds']=fold;save_json(status_path,record)
    print('ONLINE_CHECK',flush=True);online(cfg)
    print('SYNTHETIC_ROBUSTNESS',flush=True);synthetic(cfg)
    print('FINAL_REPORT',flush=True);report(cfg)
    record['status']='complete'
except BaseException as exc:
    record.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());raise
finally:
    record['finish']=datetime.now(timezone.utc).isoformat();save_json(status_path,record)
