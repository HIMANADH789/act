import urllib.request, urllib.parse, ssl, json, re
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Read JS to find the exact lists of symbols for each asset type
url = 'https://volare.unime.it/static/js/main.519c6748.js'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
    js = resp.read().decode('utf-8', errors='ignore')

# Look for forex, futures, ETFs lists
for m in re.finditer(r'(?:forex|futures|ETFs|stocks)[^:]*:[^\[]*(\[[^\]]+\])', js):
    print('Match:', m.group(0)[:150])

for m in re.finditer(r'\{[^{}]*symbol:[^{}]*\}', js):
    print('Symbol obj:', m.group(0)[:100])
