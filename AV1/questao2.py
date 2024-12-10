import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carregar os dados
stars = pd.read_csv('dataset/classification/star_classification_ajustado.csv')

y = stars['class']
X = stars.drop(["class"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter para numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# Funções de distância
def dist_euclidiana(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2, axis=1))


def dist_manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2), axis=1)


def dist_mahalanobis_batch(X_train, nova_amostra, inv_cov_matrix):
    diffs = X_train - nova_amostra
    distances = np.sqrt(np.sum(diffs @ inv_cov_matrix * diffs, axis=1))
    return distances

def dist_chebyshev(v1, v2):
    return np.max(np.abs(v1 - v2), axis=1)


# KNN com suporte para Mahalanobis
def knn(X_train, y_train, nova_amostra, K, dist_func, inv_cov_matrix=None):
    if dist_func == dist_mahalanobis_batch:
        distances = dist_func(X_train, nova_amostra, inv_cov_matrix)
    else:
        distances = dist_func(X_train, nova_amostra)

    k_indices = np.argsort(distances)[:K]
    k_vizinhos = y_train[k_indices]
    return np.bincount(k_vizinhos).argmax()


def calcular_acuracia(X_test, y_test, K, dist_func, covariance_matrix=None):
    acertos = 0
    for i in range(len(X_test)):
        classe_prevista = knn(X_train, y_train, X_test[i], K, dist_func, covariance_matrix)
        if classe_prevista == y_test[i]:
            acertos += 1
    return 100 * acertos / len(X_test)


# Resultados para diferentes distâncias
print("Distância Euclidiana:")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=dist_euclidiana)
print("Porcentagem de acertos:", acuracia)

print("Distância de Manhattan:")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=dist_manhattan)
print("Porcentagem de acertos:", acuracia)

# covariance matrix and its inverse
cov_matrix = np.cov(X_train.T)
alpha = 20
relaxed_cov_matrix = cov_matrix + alpha * np.eye(cov_matrix.shape[0])
inv_cov_matrix = np.linalg.inv(relaxed_cov_matrix)
print("Distância Mahalanobis (otimizada):")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=dist_mahalanobis_batch, covariance_matrix=inv_cov_matrix)
print("Porcentagem de acertos:", acuracia)


print("Distância de Chebyshev:")
acuracia = calcular_acuracia(X_test, y_test, K=7, dist_func=dist_chebyshev)
print("Porcentagem de acertos:", acuracia)
