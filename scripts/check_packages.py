import importlib
for pkg in ['torch', 'chronos', 'timesfm', 'arch', 'scipy', 'sklearn', 'statsmodels', 'gpytorch', 'transformers']:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}: {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED ({e})")

