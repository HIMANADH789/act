import argparse
import importlib.metadata
import platform
import subprocess
import sys
import psutil
from .config import ROOT,load,directories,save_json

def execute():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['download','reference','simulate','online','robustness','report','all','environment'])
    parser.add_argument('--config')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--core-only',action='store_true')
    args=parser.parse_args();cfg=load(args.config);directories()
    if args.smoke:
        cfg['windows_per_asset_cell']=2;cfg['scope']='smoke';cfg['folds']=cfg['folds'][:1]
    if args.core_only:cfg['foundation_models']=False
    import hashlib,json
    cfg['hash']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    save_json(ROOT/'results/manifests/config.json',cfg)
    if args.command in ('environment','all'):
        env=dict(python=sys.version,os=platform.platform(),memory=psutil.virtual_memory()._asdict(),
            cpu_count=psutil.cpu_count(),disk_free=psutil.disk_usage(str(ROOT)).free,
            packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
        try:
            import torch
            env['cuda']=dict(version=torch.version.cuda,available=torch.cuda.is_available(),
                name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        except Exception as e:env['cuda_error']=repr(e)
        save_json(ROOT/'results/manifests/environment.json',env)
        print(env.get('cuda',env.get('cuda_error')),flush=True)
    if args.command in ('download','all'):
        from .data_download import download
        download(cfg)
    if args.command in ('reference','all'):
        from .runner import reference
        reference(cfg)
    if args.command in ('simulate','all'):
        from .runner import simulations
        simulations(cfg)
    if args.command in ('online','all'):
        from .online_runner import online
        online(cfg)
    if args.command in ('robustness','all'):
        from .robustness import synthetic
        synthetic(cfg)
    if args.command in ('report','all'):
        from .reporting import report
        report(cfg)

def main():
    import os,time,hashlib
    from datetime import datetime,timezone
    directories();started=time.perf_counter()
    code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src/act_tsfm').rglob('*.py')))).hexdigest()
    record=dict(start=datetime.now(timezone.utc).isoformat(),arguments=sys.argv[1:],pid=os.getpid(),code_sha256=code_hash,status='running')
    path=ROOT/f'results/logs/run_{time.time_ns()}_{os.getpid()}.json'
    save_json(path,record)
    try:
        execute();record['status']='complete'
    except BaseException as exc:
        record.update(status='failed_or_interrupted',error=repr(exc));raise
    finally:
        record.update(finish=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started)
        save_json(path,record)

if __name__=='__main__':main()
