import numpy as np
from .config import ROOT,save_json
from .preprocessing import process,split,FEATURES

def validate(cfg):
    data=process(cfg);checks=[]
    assert len(data)>=6
    for asset,d in data.items():
        for train_end,cal_end,test_end in cfg['folds']:
            tr=split(d,2010,train_end);ca=split(d,train_end+1,cal_end);te=split(d,cal_end+1,test_end)
            assert tr.outcome5.max()<ca.index.min()
            assert ca.outcome5.max()<te.index.min()
            assert np.isfinite(tr[FEATURES].to_numpy()).all()
            assert (tr[['rv1','rv5']].to_numpy()>=0).all()
            checks.append(dict(asset=asset,train_end=train_end,cal_end=cal_end,test_end=test_end,
                               split_separation=True,features_finite=True,targets_nonnegative=True))
    save_json(ROOT/'results/manifests/data_validation.json',checks)
    return checks
