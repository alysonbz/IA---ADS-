import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Carregar o dataset
WineQT = pd.read_csv('./dataset/wineqt_ajustado.csv')

# Crie um DataFrame com todas as colunas, com exceção de ``quality``
X = WineQT.drop("quality", axis=1).values

# Crie um dataframe de labels com a coluna quality
y = WineQT['quality'].values

# Separando o conjunto de teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Preparar os dados de treinamento e teste
treinamento = np.hstack((X_train, y_train.reshape(-1, 1)))
teste = np.hstack((X_test, y_test.reshape(-1, 1)))

# Funções de distância
def dist_euclidiana(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def dist_manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2))

def dist_chebyshev(v1, v2):
    return np.max(np.abs(v1 - v2))

def dist_mahalanobis(v1, v2, matriz_cov):
    delta = np.array(v1) - np.array(v2)
    inv_cov = np.linalg.inv(matriz_cov)
    return np.sqrt(np.dot(np.dot(delta.T, inv_cov), delta))

# Implementação do KNN
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
    return max(set(classes), key=classes.count)

# Avaliar o KNN
def avaliar_knn(teste, treinamento, K, distancia_func, matriz_cov=None):
    acertos = 0
    for amostra in teste:
        classe = knn(treinamento, amostra, K, distancia_func, matriz_cov)
        if amostra[-1] == classe:
            acertos += 1
    return 100 * acertos / len(teste)

# Matriz de covariância para distância Mahalanobis
matriz_cov = np.cov(X_train.T)

# Avaliar para diferentes distâncias
K = 7
print("Porcentagem de acertos distância euclidiana:",
      avaliar_knn(teste, treinamento, K, dist_euclidiana))
print("Porcentagem de acertos distância manhattan:",
      avaliar_knn(teste, treinamento, K, dist_manhattan))
print("Porcentagem de acertos distância chebyshev:",
      avaliar_knn(teste, treinamento, K, dist_chebyshev))
print("Porcentagem de acertos distância mahalanobis:",
      avaliar_knn(teste, treinamento, K, dist_mahalanobis, matriz_cov))
