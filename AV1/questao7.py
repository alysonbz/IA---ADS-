import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset
dataset = pd.read_csv('Possum_Data_Adjusted.csv')

# Definir o atributo mais relevante e o alvo
X = dataset[['totlngth']]
y = dataset['age']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 2. Implementação manual de k-fold cross-validation
def k_fold_cross_validation(X, y, k=5):
    """
    Implementa o k-fold cross-validation manualmente.

    Parâmetros:
    - X: Dados de entrada
    - y: Alvo
    - k: Número de folds

    Retorna:
    - Média das métricas (RSS, MSE, RMSE, R²) ao longo dos k folds
    """
    fold_size = len(X) // k
    rss_list, mse_list, rmse_list, r2_list = [], [], [], []

    for fold in range(k):
        # Separar os índices para treino e validação
        start = fold * fold_size
        end = start + fold_size
        X_val = X[start:end]
        y_val = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        # Treinar o modelo de regressão linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Previsões no conjunto de validação
        y_pred = model.predict(X_val)

        # Cálculo manual das métricas
        rss = np.sum((y_val - y_pred) ** 2)  # Residual Sum of Squares (RSS)
        mse = np.mean((y_val - y_pred) ** 2)  # Mean Squared Error (MSE)
        rmse = np.sqrt(mse)  # Root Mean Squared Error (RMSE)
        r_squared = 1 - (rss / np.sum((y_val - np.mean(y_val)) ** 2))  # R²

        # Salvar métricas
        rss_list.append(rss)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r_squared)

    # Retornar a média das métricas
    return {
        "RSS": np.mean(rss_list),
        "MSE": np.mean(mse_list),
        "RMSE": np.mean(rmse_list),
        "R²": np.mean(r2_list)
    }


# 3. Aplicar a validação cruzada manual
k = 5  # Número de folds
metrics = k_fold_cross_validation(X_scaled, y.to_numpy(), k=k)

# 4. Exibir as métricas calculadas
print(f"Métricas (média ao longo de {k} folds):")
print(f"RSS: {metrics['RSS']}")
print(f"MSE: {metrics['MSE']}")
print(f"RMSE: {metrics['RMSE']}")
print(f"R²: {metrics['R²']}")
