from sklearn.model_selection import train_test_split
from AV1.src.utils import load_water_quality
import numpy as np
import math

waterQuality = load_water_quality()

X = waterQuality.drop(['is_safe'], axis=1).values
y = waterQuality['is_safe'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

treinamento = np.hstack((X_train, y_train.reshape(-1, 1)))
teste = np.hstack((X_test, y_test.reshape(-1, 1)))

def dist_euclidiana(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))

def dist_manhattan(v1, v2):
    return sum(abs(v1[i] - v2[i]) for i in range(len(v1)))

def dist_chebyshev(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1)))

def dist_mahalanobis(v1, v2, matriz_cov):
    delta = np.array(v1) - np.array(v2)
    inv_cov = np.linalg.inv(matriz_cov)
    return np.sqrt(np.dot(np.dot(delta.T, inv_cov), delta))

def knn(treinamento, nova_amostra, K, distancia_func, matriz_cov=None):
    dists = {}
    for i in range(len(treinamento)):
        if matriz_cov is not None and distancia_func == dist_mahalanobis:
            d = distancia_func(treinamento[i][:-1], nova_amostra[:-1], matriz_cov)
        else:
            d = distancia_func(treinamento[i][:-1], nova_amostra[:-1])
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    classes = [treinamento[i][-1] for i in k_vizinhos]
    qtd_safe = sum(1 for c in classes if c == 1.0)
    qtd_unsafe = sum(1 for c in classes if c == 0.0)

    return 1.0 if qtd_safe > qtd_unsafe else 0.0

def avaliar_knn(teste, treinamento, K, distancia_func, matriz_cov=None):
    acertos = 0
    for amostra in teste:
        classe = knn(treinamento, amostra, K, distancia_func, matriz_cov)
        if amostra[-1] == classe:
            acertos += 1
    return 100 * acertos / len(teste)

matriz_cov = np.cov(X_train.T)

K = 7
print("Porcentagem de acertos distância euclidiana:",
      avaliar_knn(teste, treinamento, K, dist_euclidiana))
print("Porcentagem de acertos distância manhattan:",
      avaliar_knn(teste, treinamento, K, dist_manhattan))
print("Porcentagem de acertos distância chebyshev:",
      avaliar_knn(teste, treinamento, K, dist_chebyshev))
print("Porcentagem de acertos distância mahalanobis:",
      avaliar_knn(teste, treinamento, K, dist_mahalanobis, matriz_cov))
