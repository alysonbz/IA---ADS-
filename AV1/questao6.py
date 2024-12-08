import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Carregar o dataset
Regression = pd.read_csv('./dataset/Regression_ajustado.csv')

# Ajustar as dimensões de X e y
X = Regression["sp500 open"].values.reshape(-1, 1)  # Transformar em array 2D
y = Regression["sp500 low"].values  # Array 1D está correto

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear
reg = LinearRegression()
reg.fit(X_train, y_train)

# Fazer predições
y_pred = reg.predict(X_test)

# Visualizar os dados e a linha de regressão
plt.scatter(X, y, alpha=0.7, label="Dados reais")
plt.plot(X_test, y_pred, color="red", label="Linha de regressão")
plt.xlabel("sp500 open")
plt.ylabel("sp500 low")
plt.legend()
plt.show()

# Calcular métricas
r_squared = reg.score(X_test, y_test)
rss = np.sum(np.square(y_test - reg.predict(X_test)))  # Ajustado para os dados de teste
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Exibir os resultados
print(f"RSS: {rss:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r_squared:.4f}")
