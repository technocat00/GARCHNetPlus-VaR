import numpy as np
import pandas as pd
import yfinance as yf


def load_returns(
    symbol: str,
    start: str,
    end: str,
    source: str = "yfinance",
    price_col: str = "Close",
    scale: float = 100.0,
) -> pd.Series:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(f"No yfinance data downloaded for symbol={symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        price = df[price_col].iloc[:, 0]
    else:
        price = df[price_col]

    price = price.dropna().astype(float)

    returns = scale * np.log(price / price.shift(1))
    returns = returns.dropna()
    returns.name = "return"

    return returns
