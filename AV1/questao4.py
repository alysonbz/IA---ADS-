import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

stars = pd.read_csv('dataset/classification/star_classification_ajustado.csv')
y = stars['class']
X = stars.drop(["class"], axis=1)
cols_to_normalize = [col for col in X.columns if col != 'redshift']

# normalização Logarítmica
X_log = X.copy()
X_log[cols_to_normalize] = np.log(X[cols_to_normalize])

X_train_log, X_test_log, y_train, y_test = train_test_split(X_log.values, y.values, test_size=0.2, random_state=42)


def dist_manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2), axis=1)


# função KNN
def knn(X_train, y_train, nova_amostra, K, dist_func):
    distances = dist_func(X_train, nova_amostra)  # Calcular distâncias
    k_indices = np.argsort(distances)[:K]  # Índices dos K vizinhos mais próximos
    k_vizinhos = y_train[k_indices]  # Classes dos K vizinhos
    return np.bincount(k_vizinhos).argmax()  # Classe mais comum


def calcular_acuracia(X_test, y_test, K, dist_func, X_train, y_train):
    acertos = 0
    for i in range(len(X_test)):
        classe_prevista = knn(X_train, y_train, X_test[i], K, dist_func)
        if classe_prevista == y_test[i]:
            acertos += 1
    return 100 * acertos / len(X_test)


# testar diferentes valores de K
ks = range(1, 21)
acuracias = []

for k in ks:
    acuracia = calcular_acuracia(X_test_log, y_test, K=k, dist_func=dist_manhattan, X_train=X_train_log,
                                 y_train=y_train)
    acuracias.append(acuracia)
    print(f"Acurácia com K = {k}: {acuracia:.2f}%")

# identificar o melhor K
melhor_k = ks[np.argmax(acuracias)]
melhor_acuracia = max(acuracias)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(ks, acuracias, marker='o', linestyle='-')
plt.title('Acurácia do KNN com Normalização Logarítmica para Diferentes Valores de K')
plt.xlabel('Valor de K')
plt.ylabel('Acurácia (%)')
plt.grid()
plt.xticks(ks)
plt.axvline(melhor_k, color='red', linestyle='--', label=f"Melhor K = {melhor_k}")
plt.legend()
plt.show()

# Exibir resultados
print(f"Melhor K: {melhor_k}")
print(f"Acurácia com Melhor K: {melhor_acuracia:.2f}%")
