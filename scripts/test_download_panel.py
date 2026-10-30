import urllib.request, urllib.parse, ssl, json, time, os
import pandas as pd

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

login_url = 'https://volare.unime.it/api/auth/token'
form_data = urllib.parse.urlencode({
    'username': 'act_research_bot@gmail.com',
    'password': 'ActResearch2026!Secure'
}).encode('utf-8')

req = urllib.request.Request(login_url, data=form_data, headers={
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
    'User-Agent': 'Mozilla/5.0'
})

with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
    token = json.loads(resp.read().decode('utf-8'))['access_token']

panel = [
    ('AAPL', 'stocks', 'Equities'),
    ('MSFT', 'stocks', 'Equities'),
    ('JPM', 'stocks', 'Equities'),
    ('XOM', 'stocks', 'Equities'),
    ('ES', 'futures', 'Futures_Index'),
    ('CL', 'futures', 'Futures_Commodity'),
    ('GC', 'futures', 'Futures_Commodity'),
    ('EURUSD', 'forex', 'Forex'),
    ('GBPUSD', 'forex', 'Forex'),
    ('USDJPY', 'forex', 'Forex'),
]

os.makedirs('data/volare_raw', exist_ok=True)
summary = []

for symbol, asset_type, category in panel:
    print(f'Checking {symbol} ({asset_type})...')
    url = f'https://volare.unime.it/api/financial-data/download?symbol={symbol}&asset_type={asset_type}'
    r = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(r, timeout=15, context=ctx) as res:
            meta = json.loads(res.read().decode('utf-8'))
            dl_url = 'https://volare.unime.it/api' + meta['download_url']
            
        fr = urllib.request.Request(dl_url, headers={'Authorization': f'Bearer {token}', 'User-Agent': 'Mozilla/5.0'})
        local_path = f'data/volare_raw/{asset_type}_{symbol}.csv'
        with urllib.request.urlopen(fr, timeout=20, context=ctx) as fres:
            data_bytes = fres.read()
            with open(local_path, 'wb') as f:
                f.write(data_bytes)
        
        df = pd.read_csv(local_path)
        min_date = df['date'].min()
        max_date = df['date'].max()
        n_obs = len(df)
        rv5_valid = df['rv5'].notna().sum()
        zero_rv5 = (df['rv5'] <= 0).sum()
        print(f'  {symbol}: {n_obs} rows, {min_date} to {max_date}, rv5_valid: {rv5_valid}, zero_or_neg: {zero_rv5}')
        summary.append({
            'symbol': symbol,
            'asset_type': asset_type,
            'category': category,
            'min_date': min_date,
            'max_date': max_date,
            'n_obs': n_obs,
            'rv5_valid': rv5_valid,
            'zero_or_neg': zero_rv5
        })
    except Exception as e:
        print(f'  Error for {symbol}: {e}')
    time.sleep(0.5)

print('\nSummary of Panel:')
print(pd.DataFrame(summary).to_string())

