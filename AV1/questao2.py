import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

stars = pd.read_csv('dataset/classification/star_classification_ajustado.csv')

y = stars['class']
X = stars.drop(["class"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# funções de distância
def dist_euclidiana(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2, axis=1))


def dist_manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2), axis=1)


def dist_minkowski(v1, v2, p):
    return np.sum(np.abs(v1 - v2) ** p, axis=1) ** (1 / p)


def dist_chebyshev(v1, v2):
    return np.max(np.abs(v1 - v2), axis=1)


# KNN implementation with NumPy
def knn(X_train, y_train, nova_amostra, K, dist_func):
    # calculate distances for all training samples
    distances = dist_func(X_train, nova_amostra)

    # get the indices of the K nearest neighbors
    k_indices = np.argsort(distances)[:K]

    # get the labels of the K nearest neighbors
    k_vizinhos = y_train[k_indices]

    # return the most common label
    return np.bincount(k_vizinhos).argmax()


def calcular_acuracia(X_test, y_test, K, dist_func):
    acertos = 0
    for i in range(len(X_test)):
        classe_prevista = knn(X_train, y_train, X_test[i], K, dist_func)
        if classe_prevista == y_test[i]:
            acertos += 1
    return 100 * acertos / len(X_test)


# resultados para diferentes distâncias
print("Distância Euclidiana:")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=dist_euclidiana)
print("Porcentagem de acertos:", acuracia)

print("Distância de Manhattan:")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=dist_manhattan)
print("Porcentagem de acertos:", acuracia)

print("Distância de Minkowski com p=3:")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=lambda v1, v2: dist_minkowski(v1, v2, p=3))
print("Porcentagem de acertos:", acuracia)

print("Distância de Chebyshev:")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=dist_chebyshev)
print("Porcentagem de acertos:", acuracia)
