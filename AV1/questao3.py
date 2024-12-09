import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

stars = pd.read_csv('dataset/classification/star_classification_ajustado.csv')
y = stars['class']
X = stars.drop(["class"], axis=1)
cols_to_normalize = [col for col in X.columns if col != 'redshift']

# normalização Logarítmica
X_log = X.copy()
X_log[cols_to_normalize] = np.log(X[cols_to_normalize])

# normalização de Média Zero e Variância Unitária
X_zscore = X.copy()
X_zscore[cols_to_normalize] = (X[cols_to_normalize] - X[cols_to_normalize].mean()) / X[cols_to_normalize].std()

X_train_log, X_test_log, y_train, y_test = train_test_split(X_log.values, y.values, test_size=0.2, random_state=42)
X_train_zscore, X_test_zscore, _, _ = train_test_split(X_zscore.values, y.values, test_size=0.2, random_state=42)


def dist_manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2), axis=1)


# função KNN
def knn(X_train, y_train, nova_amostra, K, dist_func):
    distances = dist_func(X_train, nova_amostra)
    k_indices = np.argsort(distances)[:K]
    k_vizinhos = y_train[k_indices]
    return np.bincount(k_vizinhos).argmax()


def calcular_acuracia(X_test, y_test, K, dist_func, X_train, y_train):
    acertos = 0
    for i in range(len(X_test)):
        classe_prevista = knn(X_train, y_train, X_test[i], K, dist_func)
        if classe_prevista == y_test[i]:
            acertos += 1
    return 100 * acertos / len(X_test)


# Avaliar a acurácia com normalização logarítmica
print("Com Normalização Logarítmica:")
acuracia_log = calcular_acuracia(X_test_log, y_test, K=7, dist_func=dist_manhattan, X_train=X_train_log,
                                 y_train=y_train)
print("Acurácia:", acuracia_log)

# Avaliar a acurácia com normalização de média zero e variância unitária
print("\nCom Normalização Z-Score:")
acuracia_zscore = calcular_acuracia(X_test_zscore, y_test, K=7, dist_func=dist_manhattan, X_train=X_train_zscore,
                                    y_train=y_train)
print("Acurácia:", acuracia_zscore)

# Comparação das duas abordagens
print("\nComparação:")
print(f"Acurácia Logarítmica: {acuracia_log:.2f}%")
print(f"Acurácia Z-Score: {acuracia_zscore:.2f}%")
