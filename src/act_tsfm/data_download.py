"""Provider cache. No fabricated or forward-filled histories."""
import hashlib
import time
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
from .config import ROOT, save_json

def download(cfg: dict) -> list[dict]:
    manifest = []
    for ticker in cfg['tickers']:
        path = ROOT / f'data/raw/{ticker}.csv'
        error = None
        if not path.exists():
            for attempt in range(3):
                try:
                    df = yf.download(ticker, start=cfg['start'], end=cfg['end'], auto_adjust=False,
                                     progress=False, threads=False, timeout=25)
                    if df.empty:
                        raise ValueError('Empty provider response')
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if 'Adj Close' not in df:
                        raise ValueError('Adjusted close absent')
                    df.to_csv(path)
                    error = None
                    break
                except Exception as exc:
                    error = repr(exc)
                    time.sleep(attempt + 1)
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            manifest.append(dict(ticker=ticker, rows=len(df), start=str(df.index.min()), end=str(df.index.max()),
                source='Yahoo Finance via yfinance; personal research use', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                missing=df.isna().sum().to_dict(), adequate=len(df)>3500,
                cache_mtime=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
        else:
            manifest.append(dict(ticker=ticker, adequate=False, error=error))
        print(ticker, manifest[-1], flush=True)
    save_json(ROOT/'data/manifests/download.json',manifest)
    if sum(x['adequate'] for x in manifest)<6:
        raise RuntimeError('Fewer than six assets with adequate histories; real-data experiments halted')
    return manifest
