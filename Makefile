.PHONY: install lint test train

install:
	pip install -r requirements.txt

lint:
	python -m compileall src

test:
	pytest -q

train:
	python -m src.dota_draft.modeling.train
