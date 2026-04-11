# preprocessing.py
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cleaned_dataset(csv_path, date_col):
    """
    Load the already-saved cleaned CSV from your preprocessing notebook.
    No scaling is done here because the saved file is already scaled.
    """
    df = pd.read_csv(csv_path)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)

    # Keep numeric columns only for modelling
    df = df.select_dtypes(include=[np.number]).copy()

    if df.empty:
        raise ValueError(f"No numeric columns found in {csv_path}")

    return df


def infer_target_col(df, dataset_name=None, target_col=None):
    """
    Chooses a default target column if you do not pass one.
    """
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"target_col='{target_col}' not found. Available columns: {list(df.columns)}"
            )
        return target_col

    if dataset_name == "apple" and "Close" in df.columns:
        return "Close"

    weather_candidates = [
        "T (degC)",
        "T",
        "temp",
        "temperature",
        "temperature_2m",
        "Air temperature"
    ]
    for col in weather_candidates:
        if col in df.columns:
            return col

    # Fallback: first numeric column
    return df.columns[0]


def split_by_time(df, train_ratio=0.7, val_ratio=0.15):
    """
    Time-based split: train / val / test
    """
    if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("Need 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def create_sequences_xy(features, target, window):
    """
    features: numpy array of shape [N, num_features]
    target:   numpy array of shape [N, 1]
    returns:
        X: [num_samples, window, num_features]
        y: [num_samples, 1]
    """
    X, y = [], []

    for i in range(len(features) - window):
        X.append(features[i:i + window])
        y.append(target[i + window])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y


def make_dataloaders(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size=64,
    seed=42
):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


def prepare_timeseries_from_cleaned(
    csv_path,
    date_col,
    window,
    dataset_name=None,
    target_col=None,
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=64,
    seed=42
):
    """
    Full pipeline from saved cleaned CSV -> dataloaders
    """
    df = load_cleaned_dataset(csv_path, date_col=date_col)
    target_col = infer_target_col(df, dataset_name=dataset_name, target_col=target_col)

    feature_cols = list(df.columns)

    train_df, val_df, test_df = split_by_time(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    X_train, y_train = create_sequences_xy(
        train_df[feature_cols].values,
        train_df[[target_col]].values,
        window
    )
    X_val, y_val = create_sequences_xy(
        val_df[feature_cols].values,
        val_df[[target_col]].values,
        window
    )
    X_test, y_test = create_sequences_xy(
        test_df[feature_cols].values,
        test_df[[target_col]].values,
        window
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=batch_size,
        seed=seed
    )

    return {
        "df": df,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "window": window,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_train_shape": X_train.shape,
        "y_train_shape": y_train.shape,
        "X_val_shape": X_val.shape,
        "y_val_shape": y_val.shape,
        "X_test_shape": X_test.shape,
        "y_test_shape": y_test.shape,
    }   