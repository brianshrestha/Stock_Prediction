"""
feature_utils.py
Utility functions for the HW6 Streamlit deployment.
Provides extract_features() which returns a DataFrame whose columns match
MODEL_INFO['keys'] in StreamlitApp_HW6.py:
    ['ADBE', 'MSFT', 'JPM', 'sentiment_textblob']
"""

import os
import pandas as pd
import numpy as np


def extract_features(
    sentiment_path: str = "DataWithSentimentsResults_HW.csv",
    ticker: str = "NFLX",
) -> pd.DataFrame:
    """
    Load the sentiment dataset and return a representative one-row DataFrame
    with the four feature columns used by the deployed model:
        ADBE, MSFT, JPM  -> average daily textblob sentiment for those tickers
        sentiment_textblob -> average daily textblob sentiment for NFLX itself

    This is used by the Streamlit app to pre-fill default values and provide
    a base DataFrame for the SHAP explainer preprocessing step.

    Parameters
    ----------
    sentiment_path : str
        Path to DataWithSentimentsResults_HW.csv  (pipe-delimited).
    ticker : str
        Target ticker (default 'NFLX').

    Returns
    -------
    pd.DataFrame  with one row and columns ['ADBE','MSFT','JPM','sentiment_textblob']
    """
    # Resolve path relative to this file's directory when running on Streamlit Cloud
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, sentiment_path)

    try:
        sent_dataset = pd.read_csv(full_path, sep="|")

        # Pivot other-ticker sentiments
        other_tickers = ["ADBE", "MSFT", "JPM"]
        pivot = (
            sent_dataset[sent_dataset["ticker"].isin(other_tickers)]
            .groupby("ticker")["sentiment_textblob"]
            .mean()
        )
        row = {t: float(pivot.get(t, 0.0)) for t in other_tickers}

        # Target-ticker own sentiment
        target_sent = (
            sent_dataset[sent_dataset["ticker"] == ticker]["sentiment_textblob"].mean()
        )
        row["sentiment_textblob"] = float(target_sent) if not np.isnan(target_sent) else 0.0

    except Exception:
        # Fallback: return neutral values so the app still loads
        row = {"ADBE": 0.0, "MSFT": 0.0, "JPM": 0.0, "sentiment_textblob": 0.0}

    return pd.DataFrame([row])


def get_bitcoin_historical_prices():
    """Stub kept for backward compatibility with the template import."""
    return pd.DataFrame()
