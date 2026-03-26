import pandas as pd
import numpy as np
from pathlib import Path
from predictakit.core import logistic_regression
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
y = (df['FTR'] == 'H').astype(int).values

#Split
X_train = X[:300]
y_train = y[:300]

X_test = X[300:]
y_test = y[300:]

# Train
model = logistic_regression.LogisticRegression()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = np.mean(test_predictions == y_test)
print(f"Accuracy: {accuracy:.2%}")
home_win_rate = np.mean(y_test)
print(f"Home win rate in test set: {home_win_rate:.2%}")

test_probas = model.predict_proba(X_test)
print(f"Min probability: {test_probas.min():.4f}")
print(f"Max probability: {test_probas.max():.4f}")
print(f"Mean probability: {test_probas.mean():.4f}")