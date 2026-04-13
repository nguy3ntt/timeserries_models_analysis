import os
import sys
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# make imports work from root, utils, or notebooks
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) in {'utils', 'notebooks'}:
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from utils.preprocessing import set_seed, prepare_timeseries_from_cleaned
from models.mlp import MLP
from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.transformer import TransformerModel



DATASETS = {
    "apple": {
        "csv_path": "data/cleaned/apple_cleaned.csv",
        "date_col": "Date",
        "window": 30,
        "target_col": "log_return",
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-3,
        "patience": 10,
    },
    "weather": {
        "csv_path": "data/cleaned/weather_cleaned.csv",
        "date_col": "date",
        "window": 144,
        "target_col": "T",
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-3,
        "patience": 10,
    },
}

MODELS = ["mlp", "rnn", "lstm", "transformer"]


def build_model(model_name, input_size, seq_len):
    if model_name == "mlp":
        return MLP(input_size=input_size, seq_len=seq_len)
    if model_name == "rnn":
        return RNNModel(input_size=input_size, hidden_size=64, num_layers=1)
    if model_name == "lstm":
        return LSTMModel(input_size=input_size, hidden_size=64, num_layers=1)
    if model_name == "transformer":
        return TransformerModel(
            input_size=input_size,
            seq_len=seq_len,
            d_model=64,
            nhead=4,
            num_layers=2
        )
    raise ValueError(f"unknown model: {model_name}")


def to_2d(x):
    if x.ndim == 1:
        return x.unsqueeze(1)
    return x


def train_model(model, train_loader, val_loader, epochs, lr, patience, device):
    # use mse for one-step forecasting
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    model.to(device)

    for epoch in range(epochs):
        # train for one full epoch
        model.train()
        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = to_2d(model(X_batch))
            y_batch = to_2d(y_batch)

            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * X_batch.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        history["train_loss"].append(avg_train_loss)

        # check validation after each epoch
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = to_2d(model(X_batch))
                y_batch = to_2d(y_batch)
                loss = criterion(preds, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)

        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    model.load_state_dict(best_state)
    return model, history


def predict(model, data_loader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds = to_2d(model(X_batch))
            all_preds.append(preds.cpu().numpy())
            all_true.append(to_2d(y_batch).cpu().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_true)
    return y_true, y_pred


def get_metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


def save_loss_plot(history, save_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_prediction_plot(y_true, y_pred, save_path, title, n_points=300):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    n_points = min(n_points, len(y_true))

    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:n_points], label="True")
    plt.plot(y_pred[:n_points], label="Predicted")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_dataset(dataset_name, config, assets_dir, device):
    print(f"\n{'=' * 14} {dataset_name.upper()} {'=' * 14}")

    data = prepare_timeseries_from_cleaned(
        csv_path=config["csv_path"],
        date_col=config["date_col"],
        window=config["window"],
        dataset_name=dataset_name,
        target_col=config["target_col"],
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=config["batch_size"],
        seed=42,
    )

    print("Target column:", data["target_col"])
    print("Feature count:", data["n_features"])
    print("X_train:", data["X_train_shape"], "y_train:", data["y_train_shape"])
    print("X_val:  ", data["X_val_shape"], "y_val:  ", data["y_val_shape"])
    print("X_test: ", data["X_test_shape"], "y_test: ", data["y_test_shape"])

    dataset_dir = assets_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name in MODELS:
        print(f"\n----- {dataset_name.upper()} | {model_name.upper()} -----")
        set_seed(42)

        model = build_model(
            model_name=model_name,
            input_size=data["n_features"],
            seq_len=config["window"],
        )

        model, history = train_model(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            epochs=config["epochs"],
            lr=config["lr"],
            patience=config["patience"],
            device=device,
        )

        y_true, y_pred = predict(model, data["test_loader"], device)
        rmse, mae = get_metrics(y_true, y_pred)

        print(f"{dataset_name} | {model_name}")
        print("  y_true shape:", y_true.shape, "y_pred shape:", y_pred.shape)
        print("  y_true min/max:", float(y_true.min()), float(y_true.max()))
        print("  y_pred min/max:", float(y_pred.min()), float(y_pred.max()))
        print("  first 5 y_true:", y_true[:5].reshape(-1))
        print("  first 5 y_pred:", y_pred[:5].reshape(-1))
        print(f"Test RMSE: {rmse:.6f} | Test MAE: {mae:.6f}")

        save_loss_plot(
            history,
            dataset_dir / f"{model_name}_loss.png",
            f"{dataset_name.upper()} - {model_name} Loss",
        )
        save_prediction_plot(
            y_true,
            y_pred,
            dataset_dir / f"{model_name}_predictions.png",
            f"{dataset_name.upper()} - {model_name} Predictions",
        )

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "rmse": rmse,
            "mae": mae,
            "best_val_loss": min(history["val_loss"]),
            "epochs_ran": len(history["val_loss"]),
        })

    return results


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    all_results = []
    for dataset_name, config in DATASETS.items():
        results = run_dataset(dataset_name, config, assets_dir, device)
        all_results.extend(results)

    # keep one summary csv for easy comparison
    summary_path = assets_dir / "metrics_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("dataset,model,rmse,mae,best_val_loss,epochs_ran\n")
        for row in all_results:
            f.write(
                f"{row['dataset']},{row['model']},{row['rmse']},{row['mae']},{row['best_val_loss']},{row['epochs_ran']}\n"
            )

    print("\nSaved plots to:", assets_dir.resolve())
    print("Saved summary to:", summary_path.resolve())