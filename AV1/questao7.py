import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# 1. Carregar o dataset
file_path = os.path.join(os.getcwd(), 'Life Expectancy Data.csv')  # Caminho dinâmico para o arquivo
dataset = pd.read_csv(file_path)

# 2. Pré-processamento (remover NaN, colunas desnecessárias)
dataset.columns = dataset.columns.str.strip()  # Remover espaços nos nomes das colunas
dataset = dataset.select_dtypes(include=[np.number])  # Selecionar somente colunas numéricas
dataset = dataset.fillna(dataset.mean())  # Preencher NaN com a média das colunas

# 3. Definir variável dependente (y) e independente (X)
X = dataset.drop(columns=['Life expectancy'])
y = dataset['Life expectancy']

# 4. Implementação manual de K-fold (por exemplo, K=5)
k = 5
fold_size = len(X) // k
rss_list = []
mse_list = []
rmse_list = []
r2_list = []

# K-fold cross-validation
for i in range(k):
    # Dividir os dados em treino e teste
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < k - 1 else len(X)

    X_train = pd.concat([X[:start_idx], X[end_idx:]])
    y_train = pd.concat([y[:start_idx], y[end_idx:]])
    X_test = X[start_idx:end_idx]
    y_test = y[start_idx:end_idx]

    # 5. Treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Fazer previsões
    y_pred = model.predict(X_test)

    # 7. Calcular métricas manualmente
    residuals = y_test - y_pred  # Resíduos

    # RSS (Residual Sum of Squares)
    rss = np.sum(residuals ** 2)
    rss_list.append(rss)

    # MSE (Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    rmse_list.append(rmse)

    # R² (Coeficiente de Determinação)
    r2 = r2_score(y_test, y_pred)
    r2_list.append(r2)

# 8. Exibir as métricas médias para os K folds
print(f"Média de RSS: {np.mean(rss_list)}")
print(f"Média de MSE: {np.mean(mse_list)}")
print(f"Média de RMSE: {np.mean(rmse_list)}")
print(f"Média de R²: {np.mean(r2_list)}")

# 9. Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(range(k), rss_list, label='RSS')
plt.plot(range(k), mse_list, label='MSE')
plt.plot(range(k), rmse_list, label='RMSE')
plt.plot(range(k), r2_list, label='R²')
plt.xlabel('Fold')
plt.ylabel('Métricas')
plt.title('Métricas de desempenho de regressão linear com K-fold Cross-validation')
plt.legend()
plt.show()
