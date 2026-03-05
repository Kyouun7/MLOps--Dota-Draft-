"""Feature engineering routines."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Return features DataFrame for modeling.

	This baseline keeps all columns unchanged and can be extended in EDA.
	"""
	return df.copy()

