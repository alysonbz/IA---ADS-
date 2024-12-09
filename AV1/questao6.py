import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# 1. Carregar o dataset ajustado
file_path = os.path.join(os.getcwd(), 'life_expotancy_data_ajustado.csv')  # Caminho dinâmico para o arquivo ajustado
dataset = pd.read_csv(file_path)

# 2. Verificar os dados
print(dataset.head())
print(dataset.info())

# 3. Selecionar o atributo mais relevante calculado na questão 5 (Exemplo: 'GDP')
# Aqui estamos assumindo que 'GDP' foi considerado o atributo mais relevante da questão 5.
X_regression = dataset[['GDP']]  # Atributo mais relevante
y_regression = dataset['Life expectancy']  # A variável alvo

# 4. Criar o modelo de regressão linear
model = LinearRegression()
model.fit(X_regression, y_regression)

# 5. Realizar a previsão
y_pred = model.predict(X_regression)

# 6. Calcular as métricas
# RSS - Residual Sum of Squares
rss = np.sum((y_regression - y_pred) ** 2)
# MSE - Mean Squared Error
mse = mean_squared_error(y_regression, y_pred)
# RMSE - Root Mean Squared Error
rmse = np.sqrt(mse)
# R² - Coeficiente de determinação
r_squared = r2_score(y_regression, y_pred)

# Exibir os valores das métricas
print(f'RSS: {rss}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r_squared}')

# 7. Plotar o gráfico da reta de regressão com a nuvem de pontos
plt.figure(figsize=(10, 6))
plt.scatter(X_regression, y_regression, color='blue', label='Nuvem de pontos')
plt.plot(X_regression, y_pred, color='red', label='Reta de regressão')
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.title('Reta de Regressão Linear: GDP vs Life Expectancy')
plt.legend()
plt.show()
