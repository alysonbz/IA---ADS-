import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

dataframe = pd.read_csv('datasets/bodyfat.csv')

X = dataframe[["Abdomen"]].values
y = dataframe["BodyFat"].values

X_log = np.log1p(X)

def kfold_cross_validation(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"RSS": [], "MSE": [], "RMSE": [], "R_squared": []}

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        rss = np.sum((y_test - y_pred) ** 2)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r_squared = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        metrics["RSS"].append(rss)
        metrics["MSE"].append(mse)
        metrics["RMSE"].append(rmse)
        metrics["R_squared"].append(r_squared)

    return {metric: np.mean(values) for metric, values in metrics.items()}


n_splits = 10
results = kfold_cross_validation(X_log, y, n_splits=n_splits)

print(f"Resultados de Regressão Linear com K-Fold Cross-Validation (n_splits={n_splits}):")
print(f"RSS Médio: {results['RSS']:.2f}")
print(f"MSE Médio: {results['MSE']:.2f}")
print(f"RMSE Médio: {results['RMSE']:.2f}")
print(f"R² Médio: {results['R_squared']:.4f}")
