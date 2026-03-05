"""Model inference entrypoint."""

import pandas as pd


def predict(df: pd.DataFrame) -> pd.Series:
	"""Generate a naive baseline prediction.

	Returns zeros to keep the pipeline executable before model selection.
	"""
	return pd.Series([0] * len(df), index=df.index, name="prediction")

