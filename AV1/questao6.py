import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.utils import Smart_Watch_Atualizado

Relogio_Inteligente = Smart_Watch_Atualizado()

#atributo mais relevante (Model ou SO)
atributo_relevante = "Model"
X = Relogio_Inteligente[[atributo_relevante]]
y = Relogio_Inteligente["Battery Life (days)"]  # Meu Alvo

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Calcular métricas de desempenho
# RSS (Residual Sum of Squares)
rss = np.sum((y_test - y_pred) ** 2)

# MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# R² (Coeficiente de Determinação)
r2 = modelo.score(X_test, y_test)

# Exibir os resultados
print(f"RSS: {rss}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Gráfico da reta de regressão e nuvem de pontos
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label="Valores Reais", alpha=0.7)
plt.plot(X_test, y_pred, color='red', label="Reta de Regressão")
plt.title(f"Regressão Linear com '{atributo_relevante}'")
plt.xlabel(atributo_relevante)
plt.ylabel("Battery Life (days)")
plt.legend()
plt.grid(True)
plt.show()
