import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Carregar o dataset
data = pd.read_csv(r'C:\Users\pinheiroiwnl\Downloads\prova_GUI\prova_GUI\king_county_preprocessed.csv')

# 2. Selecionar o atributo mais relevante
correlations = data.corr()['price'].sort_values(ascending=False)
most_relevant_feature = correlations.index[1]

X = data[[most_relevant_feature]].values
y = data['price'].values

# 3. Implementação manual do k-fold cross-validation
def kfold_cross_validation(X, y, k=5):
    n = len(X)
    fold_size = n // k
    metrics = []

    for i in range(k):
        # Separar os índices para treino e validação
        start = i * fold_size
        end = start + fold_size if i != k - 1 else n

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        X_val = X[start:end]
        y_val = y[start:end]

        # Treinar o modelo
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_val)

        # Calcular métricas manualmente
        rss = np.sum((y_val - y_pred) ** 2)
        mse = rss / len(y_val)
        rmse = np.sqrt(mse)
        ss_total = np.sum((y_val - np.mean(y_val)) ** 2)
        r_squared = 1 - (rss / ss_total)

        metrics.append({"RSS": rss, "MSE": mse, "RMSE": rmse, "R^2": r_squared})

    return metrics

# 4. Avaliar o modelo
k = 5
results = kfold_cross_validation(X, y, k=k)

# Exibir métricas para cada fold
for i, result in enumerate(results):
    print(f"Fold {i + 1}:")
    print(f"  RSS: {result['RSS']:.2f}")
    print(f"  MSE: {result['MSE']:.2f}")
    print(f"  RMSE: {result['RMSE']:.2f}")
    print(f"  R^2: {result['R^2']:.4f}")

# Calcular a média das métricas
mean_metrics = {key: np.mean([res[key] for res in results]) for key in results[0]}
print("\nMétricas médias:")
print(f"  RSS: {mean_metrics['RSS']:.2f}")
print(f"  MSE: {mean_metrics['MSE']:.2f}")
print(f"  RMSE: {mean_metrics['RMSE']:.2f}")
print(f"  R^2: {mean_metrics['R^2']:.4f}")