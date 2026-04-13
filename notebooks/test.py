import pandas as pd

df = pd.read_csv("C:/Users/ADMIN/Downloads/timeserries_models_analysis/data/cleaned/apple_cleaned.csv")
print(df.columns.tolist())
df.head(10)