import chronos
print('Chronos dir:', [x for x in dir(chronos) if not x.startswith('_')])
try:
    from chronos import ChronosPipeline
    print('ChronosPipeline available')
except Exception as e:
    print('ChronosPipeline err:', e)

try:
    import timesfm
    print('TimesFM dir:', [x for x in dir(timesfm) if not x.startswith('_')])
except Exception as e:
    print('TimesFM err:', e)

