import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [9, 10, 11, 12]}
df = pd.DataFrame(data)

print("Original Data:\n", df)

# Handle missing values
df_filled = df.fillna(df.mean())
print("After filling missing values:\n", df_filled)

# Min-Max Normalization
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_filled)
print("Normalized Data:\n", pd.DataFrame(scaled, columns=df.columns))
