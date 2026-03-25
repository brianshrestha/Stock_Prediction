import os
import pandas as pd

PAIR_KEYS = ["EMR", "GOOG"]

def extract_features_pair(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load historical raw price data for the deployed pair model.
    Returns the raw pair columns expected by the pipeline before
    PairFeatureEngineer creates rolling features.
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        csv_path = os.path.join(project_root, "Portfolio", "pair_base_history.csv")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in PAIR_KEYS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in pair_base_history.csv: {missing_cols}")

    df = df[PAIR_KEYS].copy().dropna().reset_index(drop=True)
    return df
