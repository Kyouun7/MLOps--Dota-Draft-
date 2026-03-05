"""Model training entrypoint."""

import pandas as pd


def train_model(df: pd.DataFrame) -> dict:
	"""Train a baseline model.

	Placeholder implementation to bootstrap project structure.
	"""
	return {"rows": len(df), "status": "baseline-ready"}

