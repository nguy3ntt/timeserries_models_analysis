# run_experiments.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from utils.preprocessing import set_seed, prepare_timeseries_from_cleaned
from utils.train import train_model, predict, regression_metrics


# --------------------------------------------------
# Reference model
# --------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # [batch, window, hidden]
        last_out = out[:, -1, :]       # [batch, hidden]
        out = self.fc(last_out)        # [batch, 1]
        return out


def plot_history(history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()


def run_one_experiment(name, csv_path, date_col, window, target_col=None):
    print(f"\n========== {name.upper()} ==========")

    data = prepare_timeseries_from_cleaned(
        csv_path=csv_path,
        date_col=date_col,
        window=window,
        dataset_name=name,
        target_col=target_col,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=64,
        seed=42
    )

    print("Target column:", data["target_col"])
    print("Feature count:", data["n_features"])
    print("X_train:", data["X_train_shape"], "y_train:", data["y_train_shape"])
    print("X_val:  ", data["X_val_shape"], "y_val:  ", data["y_val_shape"])
    print("X_test: ", data["X_test_shape"], "y_test: ", data["y_test_shape"])

    model = LSTMRegressor(input_size=data["n_features"])

    model, history = train_model(
        model=model,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        epochs=50,
        lr=1e-3,
        patience=10
    )

    preds, y_true = predict(model, data["test_loader"])
    metrics = regression_metrics(y_true, preds)

    print("Test metrics:", metrics)
    plot_history(history, f"{name.capitalize()} Training History")

    return model, history, metrics


if __name__ == "__main__":
    set_seed(42)

    # Apple: cleaned CSV saved from your notebook
    apple_model, apple_history, apple_metrics = run_one_experiment(
        name="apple",
        csv_path="data/cleaned/apple_cleaned.csv",
        date_col="Date",
        window=30,
        target_col="Close"
    )

    # Weather: use your saved cleaned CSV
    # If auto-detect picks the wrong column, set target_col explicitly.
    weather_model, weather_history, weather_metrics = run_one_experiment(
        name="weather",
        csv_path="data/cleaned/weather_cleaned.csv",
        date_col="date",
        window=144,
        target_col=None
        # Example if needed:
        # target_col="T (degC)"
    )