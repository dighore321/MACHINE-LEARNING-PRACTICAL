# Practical 2: Basic Statistics and Data Preprocessing using pandas and sklearn

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# Sample dataset
data = {'Marks': [50, 60, 70, 80, 90, 100]}
df = pd.DataFrame(data)

print("Original Data:\n", df)

# --- Basic Statistics ---
print("\n--- Basic Statistics ---")
print("Mean:", df['Marks'].mean())
print("Median:", df['Marks'].median())
print("Mode:", df['Marks'].mode()[0])
print("Variance:", df['Marks'].var())
print("Standard Deviation:", df['Marks'].std())

# --- Data Normalization (Min-Max Scaling) ---
scaler = MinMaxScaler()
df['MinMax_Scaled'] = scaler.fit_transform(df[['Marks']])
print("\nMin-Max Normalized Data:\n", df[['Marks', 'MinMax_Scaled']])

# --- Z-score Standardization ---
z_scores = stats.zscore(df['Marks'])
df['Z_Score'] = z_scores
print("\nZ-Score Standardized Data:\n", df[['Marks', 'Z_Score']])
