import datetime
import pandas as pd
import yfinance as yf


PAIR_KEYS = ["EMR", "GOOG"]


def extract_features_pair(
    tickers: list[str] | None = None,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Download historical adjusted close prices for the deployed pair model.

    Returns a dataframe with raw price columns only, which is what the
    deployed pipeline expects before PairFeatureEngineer creates rolling features.
    """
    if tickers is None:
        tickers = PAIR_KEYS

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days)

    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError("No price data downloaded from yfinance.")

    # Use Adj Close if available, otherwise Close
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            raise ValueError("Expected 'Adj Close' or 'Close' in downloaded data.")
    else:
        # Single ticker fallback
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers

    prices = prices[tickers].copy()
    prices = prices.dropna()
    prices.index.name = "Date"

    return prices.reset_index(drop=True)
