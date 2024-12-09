import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


file_path = 'Lenovo_ajustado.csv'

data = pd.read_csv(file_path)
print(data.head())

#atributo mais relevante (X) e o atributo alvo (y)
X = data[['High']]
y = data['Close']

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_normalized, y)

#fazer previsões
y_pred = model.predict(X_normalized)

print(f"Coeficiente: {model.coef_[0]}")
print(f"Intercepto: {model.intercept_}")


residuals = y - y_pred

RSS = np.sum(residuals ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

print(f"RSS: {RSS}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R_squared: {R_squared}")

plt.figure(figsize=(10, 6))

#nuvem de pontos
plt.scatter(X_normalized, y, color='blue', alpha=0.6, label="Dados reais")

# Reta de regressão
plt.plot(X_normalized, y_pred, color='red', label="Reta de Regressão")

plt.xlabel("Open")
plt.ylabel("Close")
plt.title("Regressão Linear - Reta e Nuvem de Pontos")
plt.legend()
plt.show()
