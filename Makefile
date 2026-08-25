PYTHON ?= .venv/bin/python

.PHONY: health test eval demo dashboard

health:
	$(PYTHON) -m rag_supply_chain.health

test:
	$(PYTHON) -m pytest -q

eval:
	$(PYTHON) -m rag_supply_chain.eval.cli run --output eval/results-current.json

demo:
	$(PYTHON) -m rag_supply_chain.demo

dashboard:
	$(PYTHON) -m rag_supply_chain.dashboard
