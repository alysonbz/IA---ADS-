import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset
dataset = pd.read_csv('Possum_Data_Adjusted.csv')

# 2. Identificar o atributo mais relevante, que já foi determinado na questão anterior
X = dataset[['totlngth']]
y = dataset['age']

# 3. Normalizar os dados (utilizando StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Criar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Previsões usando o modelo treinado
y_pred = model.predict(X_test)

# 7. Calcular as métricas de erro
rss = np.sum((y_test - y_pred) ** 2)  # Residual Sum of Squares (RSS)
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error (MSE)
rmse = np.sqrt(mse)  # Root Mean Squared Error (RMSE)
r_squared = r2_score(y_test, y_pred)

# 8. Exibir as métricas
print(f"RSS: {rss}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r_squared}")

# 9. Plotar o gráfico da reta de regressão e a nuvem de pontos
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados reais')  # Nuvem de pontos reais
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Reta de regressão')  # Reta de regressão
plt.title('Regressão Linear: Atributo "totlngth" vs "age"')
plt.xlabel('totlngth')
plt.ylabel('age')
plt.legend()
plt.show()
