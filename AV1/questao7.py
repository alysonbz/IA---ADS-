#Utilizando kfold e cross-validation faça uma regressão linear. Utilize uma implementação manual do kfold e cross-validation.
# calule as métricas RSS, MSE, RMSE e R_squared que também devem ser implementadas manualmente.
# NAO CONCLUIDA



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from src.utils import Smart_Watch_Atualizado

Relogio_Inteligente = Smart_Watch_Atualizado()

X = Relogio_Inteligente.drop(columns=["Battery Life (days)"])
y = Relogio_Inteligente["Battery Life (days)"]


def k_fold_cross_validation(X, y, k=5):
    data = pd.concat([X, y], axis=1)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    fold_size = len(data) // k
    metrics = {
        "RSS": [],
        "MSE": [],
        "RMSE": [],
        "R2": []
    }

    for fold in range(k):
        # Separando dados para o fold atual
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold != k - 1 else len(data)
        test_data = data.iloc[start:end]
        train_data = data.drop(data.index[start:end])

        X_train = train_data.drop(columns=["Battery Life (days)"])
        y_train = train_data["Battery Life (days)"]
        X_test = test_data.drop(columns=["Battery Life (days)"])
        y_test = test_data["Battery Life (days)"]

        # Treinando o modelo de regressão linear
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Fazendo previsões
        y_pred = modelo.predict(X_test)

        # Calculando as métricas
        rss = np.sum((y_test - y_pred) ** 2)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = modelo.score(X_test, y_test)

        metrics["RSS"].append(rss)
        metrics["MSE"].append(mse)
        metrics["RMSE"].append(rmse)
        metrics["R2"].append(r2)

    # Calculando a média das métricas
    avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}

    return avg_metrics


# Rodando o K-Fold Cross Validation (lembrar: esta dando erro nessa linha, olhar argumentos)
avg_metrics = k_fold_cross_validation(X, y, k=5)

print("Métricas médias após K-Fold Cross Validation:")
print(f"RSS Medio: {avg_metrics['RSS']}")
print(f"MSE Medio: {avg_metrics['MSE']}")
print(f"RMSE Medio: {avg_metrics['RMSE']}")
print(f"R2 Medio: {avg_metrics['R2']}")
