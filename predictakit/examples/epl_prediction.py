import pandas as pd
import numpy as np
from pathlib import Path
from predictakit.core import linear_regression
"""
HS / AS — home/away shots
HST / AST — home/away shots on target
HC / AC — home/away corners
HF / AF — home/away fouls
FTHG / FTAG — full-time home/away goals (we'll use these carefully)
FTR — full-time result

E0.csv - EPL Data for the 2024-2025 season
"""

data_file_path = Path(__file__).parent / "../data/samples/E0.csv"
df = pd.read_csv(data_file_path)

#Features
feature_columns = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC']
X = df[feature_columns].values

#Target
y = (df['FTHG'] - df['FTAG']).values

#Split
X_train = X[:300]
y_train = y[:300]

X_test = X[300:]
y_test = y[300:]

# Train
model = linear_regression.LinearRegression()
model.fit(X_train, y_train, method="normal")

# predict on training data
train_predictions = model.predict(X_train)
print("Train MSE:", model.mse(train_predictions, y_train))

test_predictions = model.predict(X_test)
print("Test MSE:", model.mse(test_predictions, y_test))