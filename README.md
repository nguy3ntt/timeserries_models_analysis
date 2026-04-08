timesseries-comparison/
│
├── data/
│   ├── raw/                  
│   │   ├── weather.csv
│   │   ├── apple_stock.csv
│   │
│   ├── processed/            
│       ├── weather_cleaned.csv
│       ├── apple_cleaned.csv
│
├── notebooks/
│   ├── weather_preprocessing.ipynb
│   ├── apple_preprocessing.ipynb
│   ├── eda_weather.ipynb
│   ├── eda_apple.ipynb
│
├── utils/
│   ├── preprocessing.py      
│
├── models/
│   ├── lstm_model.py
│   ├── transformer_model.py
│
├── train.py                  
├── evaluate.py               
│
├── outputs/
│   ├── plots/
│   ├── predictions/
│
├── README.md
└── requirements.txt