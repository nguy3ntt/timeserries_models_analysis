
# Time Series Model Comparison: MLP vs RNN vs LSTM vs Transformer
Author: Tin Trung Nguyen
## Overview

This project investigates and compares the performance of different machine learning models for time series forecasting. The main objective is to evaluate how well different architectures capture temporal patterns across datasets with very different characteristics.

The models compared in this project are:
- Multi-Layer Perceptron (MLP)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transformer

Two datasets are used:
- [Apple stock data (financial time series)](https://www.kaggle.com/datasets/varpit94/apple-stock-data-updated-till-22jun2021)
- [Weather time series (structured environmental data)](https://www.kaggle.com/datasets/varpit94/apple-stock-data-updated-till-22jun2021)

The project focuses on understanding:
- The importance of preprocessing and target transformation
- The impact of temporal structure on model performance
- Differences between simple and sequence-based models

---

## Project Structure


```
timeseries_models_analysis/
│
├── data/
│ ├── raw/
│ ├── cleaned/
│
├── notebooks/
│ ├── preprocessing_apple.ipynb
│ ├── preprocessing_weather.ipynb
│ ├── comparison_models.ipynb
│ ├── comparison_preds.ipynb
│
├── utils/
│ ├── preprocessing.py
│ ├── train.py
│
├── models/
│ ├── mlp.py
│ ├── rnn.py
│ ├── lstm.py
│ ├── transformer.py
│
├── assets/
│ ├── plots/
│ ├── metrics_summary.csv
│
└── README.md
```

---

## Methodology

### 1. Data Preprocessing

#### Weather Dataset
- Cleaned and resampled time series
- Standard scaling applied
- Predicting future temperature values

#### Apple Dataset
- Original stock price data contains strong upward trend
- Target transformed into **log returns**:

\[
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
\]

- This removes non-stationarity and stabilizes the series
- Allows models to focus on **short-term fluctuations**

---

### 2. Time Series Framing

- Sliding window approach used
- Input: previous `window` timesteps
- Output: next timestep

Example:
X = [t-30, ..., t-1]
y = t


- Data split chronologically:
  - Train
  - Validation
  - Test

---

### 3. Models

#### MLP
- Fully connected network
- Input flattened (no temporal awareness)
- Baseline model

#### RNN
- Basic recurrent structure
- Captures short-term dependencies

#### LSTM
- Gated recurrent network
- Handles longer-term dependencies
- More stable than vanilla RNN

#### Transformer
- Attention-based model
- Learns relationships across the entire sequence
- No recurrence

---

## Results

### Apple Dataset (Log Return)

| Model        | RMSE  | MAE   |
|--------------|------|------|
| Transformer  | **0.5699** | **0.3865** |
| LSTM         | 0.5753 | 0.3997 |
| RNN          | 0.6093 | 0.4202 |
| MLP          | 1.1741 | 0.8797 |

 **Best model in this experiment:** Transformer

---

### Weather Dataset

| Model        | RMSE  | MAE   |
|--------------|------|------|
| RNN          | **0.0361** | **0.0265** |
| LSTM         | 0.0482 | 0.0383 |
| MLP          | 0.0721 | 0.0566 |
| Transformer  | 0.0846 | 0.0688 |

 **Best model in this experiment:** RNN

---

##  Key Findings

### 1. Importance of Target Transformation
- Raw stock prices are non-stationary → poor model performance
- Using **log returns** significantly improves learning
- Enables sequence models to outperform MLP

---

### 2. Sequence Models vs MLP
- MLP performs worst in most cases
- Reason: ignores temporal order
- Sequence models (RNN, LSTM, Transformer) consistently outperform

---

### 3. Dataset Dependency

| Dataset  | Best Model | Reason |
|----------|----------|--------|
| Apple    | Transformer / LSTM | Noisy, irregular, requires flexible temporal modeling |
| Weather  | RNN | Smooth, structured, simpler temporal dependencies |

---

### 4. Model Behavior

- **RNN**:
  - Strong for smooth time series
  - Best for Weather

- **LSTM**:
  - Stable and consistent
  - Strong across both datasets

- **Transformer**:
  - Best for complex, noisy patterns
  - Slightly overkill for simpler data

- **MLP**:
  - Cannot model time dependencies
  - Weakest overall

---

### 5. Prediction Characteristics

- Weather:
  - Models capture peaks and trends well
  - RNN/LSTM closely follow true series

- Apple:
  - Predictions smoother than true values
  - Extreme spikes are not captured well
  - Expected due to high noise in financial returns

---

## Limitations

- Financial time series are inherently noisy and difficult to predict
- Models tend to **underestimate extreme events**
- Transformer performance may depend on hyperparameter tuning
- Only one window size used (fixed)

---

## Possible Improvements

- Try different window sizes (10, 30, 60)
- Add lag-based features
- Use volatility-based targets
- Hyperparameter tuning (learning rate, layers, hidden size)
- Try advanced models:
  - GRU
  - Temporal Convolutional Networks (TCN)

---

## Conclusion

This project demonstrates that:

- Proper preprocessing is critical for time series forecasting
- Removing non-stationarity (e.g., using log returns) is essential for financial data
- Sequence models significantly outperform non-temporal models
- The best architecture depends on the structure of the dataset

Overall:
- **Transformer/LSTM** are best for complex, noisy data
- **RNN** is sufficient for smoother, structured time series

---

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
