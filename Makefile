.PHONY: test smoke experiment
test:
	python -m pytest -q
smoke:
	python -m act_tsfm.cli all --smoke
experiment:
	python -m act_tsfm.cli all
