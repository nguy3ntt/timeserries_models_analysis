import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# This wont be used later on, the cleaned dataset will be load directly, but this is here for reference and potential future use
def preprocess_weather(df, date_col="date"):
    """
    Clean and preprocess weather time series data
    """
    # Datetime handling
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    # Missing values
    df = df.interpolate(method="time")
    df = df.ffill()
    df = df.dropna()

    # Outlier clipping
    df = df.clip(
        lower=df.quantile(0.01),
        upper=df.quantile(0.99),
        axis=1
    )

    # Feature engineering
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df

# This wont be used later on, the cleaned dataset will be load directly, but this is here for reference and potential future use
def preprocess_apple(df, date_col="Date"):
    """
    Clean and preprocess apple stock data
    """
    # Datetime handling
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    # Missing values
    df = df.ffill()
    df = df.dropna()

    # Feature engineering
    df["price_range"] = df["High"] - df["Low"]
    df["price_change"] = df["Close"] - df["Open"]

    # Returns
    df["return"] = df["Close"].pct_change()

    # Rolling features
    df["ma_7"] = df["Close"].rolling(window=7).mean()
    df["ma_30"] = df["Close"].rolling(window=30).mean()
    df["volatility"] = df["Close"].rolling(window=7).std()

    # Drop NaN from rolling
    df = df.dropna()

    return df

def scale_data(df, scaler_path=None):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(
        scaled_values,
        index=df.index,
        columns=df.columns
    )

    # Save scaler if path provided
    if scaler_path:
        joblib.dump(scaler, scaler_path)

    return df_scaled, scaler


def load_scaler(scaler_path):
    """
    Load a saved scaler
    """
    return joblib.load(scaler_path)

def time_split(df, train_ratio=0.8):
    """
    Time-based train test split
    """
    split_idx = int(len(df) * train_ratio)
    train = df[:split_idx]
    test = df[split_idx:]
    return train, test

def create_sequences(data, window):
    """
    Convert time series into supervised learning sequences
    """
    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])

    return np.array(X), np.array(y)