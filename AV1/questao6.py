import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Funções para métricas
def compute_RSS(predictions, y):
    RSS = np.sum(np.square(y - predictions))
    return RSS

def compute_MSE(predictions, y):
    MSE = np.mean(np.square(y - predictions))
    return MSE

def compute_RMSE(predictions, y):
    RMSE = np.sqrt(compute_MSE(predictions, y))
    return RMSE

def compute_R_squared(predictions, y):
    total_variance = np.sum(np.square(y - np.mean(y)))
    explained_variance = np.sum(np.square(predictions - np.mean(y)))
    r_squared = explained_variance / total_variance
    return r_squared

merged_data = pd.read_csv("dataset/regression/merged_ferrari_tesla.csv")

# selecionar apenas os atributos necessários
X = merged_data[["High"]].values
y = merged_data["Close"].values
scaler = StandardScaler()
X_log = np.log1p(X)  # log(1 + X) para evitar log(0) nos dados
y_log = np.log1p(y)  # log(1 + y) para evitar log(0) no alvo
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# Cálculo das métricas
RSS = compute_RSS(predictions, y_test)
MSE = compute_MSE(predictions, y_test)
RMSE = compute_RMSE(predictions, y_test)
R_squared = compute_R_squared(predictions, y_test)

print(f"RSS: {RSS:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(f"R²: {R_squared:.4f}")

# Gráfico de regressão
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label="Dados reais")
plt.plot(X_test, predictions, color='red', linewidth=2, label="Reta de regressão")
plt.xlabel("High (Normalizado)")
plt.ylabel("Close (Alvo)")
plt.title("Regressão Linear: Close vs High (Normalizado)")
plt.legend()
plt.show()