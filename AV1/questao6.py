import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv('datasets/bodyfat.csv')

X = dataframe[["Abdomen"]].values
y = dataframe["BodyFat"].values

# Normalização Logarítmica
X_log = np.log1p(X)
X_train_log, X_test_log, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Regressão Linear
regressor_log = LinearRegression()
regressor_log.fit(X_train_log, y_train)
y_pred_log = regressor_log.predict(X_test_log)

rss = np.sum((y_test - y_pred_log) ** 2)  # RSS
mse = mean_squared_error(y_test, y_pred_log)  # MSE
rmse = np.sqrt(mse)  # RMSE
r_squared = r2_score(y_test, y_pred_log)  # R_square


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_log.flatten(), y=y_test, alpha=0.7, color='b', label="Dados Reais")
plt.plot(X_test_log.flatten(), y_pred_log, color='r', label="Reta de Regressão")
plt.title(f"Regressão Linear: Abdomen (Normalização Logarítmica) vs BodyFat")
plt.xlabel("Abdomen (Normalizado)")
plt.ylabel("BodyFat")
plt.legend()
plt.show()

k_values = range(1, 21)
r2_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_log, y_train)
    y_pred_knn = knn.predict(X_test_log)
    r2_scores.append(r2_score(y_test, y_pred_knn))

melhor_k = k_values[np.argmax(r2_scores)]
melhor_r2_knn = max(r2_scores)


plt.figure(figsize=(10, 6))
plt.plot(k_values, r2_scores, marker='o', linestyle='-', color='b')
plt.title("R² do KNN para diferentes valores de k")
plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("R²")
plt.xticks(k_values)
plt.grid(True)
plt.axvline(x=melhor_k, color='r', linestyle='--', label=f'Melhor k = {melhor_k}')
plt.legend()
plt.show()

print(f"Regressão Linear com Normalização Logarítmica:")
print(f"RSS: {rss:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r_squared:.4f}\n")

