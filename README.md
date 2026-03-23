time-series-comparison/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   ├── windowing.py
│   │
│   ├── models/
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── arima.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │
│   ├── utils/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── experiments.ipynb
│
├── results/
│   ├── plots/
│   ├── metrics.csv
│
├── config.yaml
├── README.md