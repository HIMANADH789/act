python -m venv --without-pip .venv
python -m pip --python .venv install -r requirements-core.txt
python -m pip --python .venv install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip --python .venv install -r requirements-models.txt
python -m pip --python .venv install -e .
