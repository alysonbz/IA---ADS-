import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

file_path = "../AV1/datasets/novo_creditcard.csv"
df = pd.read_csv(file_path)

# Função para calcular a distância Euclidiana
def dist_euclidiana(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

# Função para calcular a distância Manhattan
def dist_manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2))

# Função para calcular a distância Mahalanobis
def dist_mahalanobis(v1, v2, cov_inv):
    diff = v1 - v2
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# Função para calcular a distância Chebyshev
def dist_chebyshev(v1, v2):
    return np.max(np.abs(v1 - v2))

# Preparando os dados
class_samples = 5000
df_stratified = df.groupby('Class').sample(n=class_samples, random_state=42)

X = df_stratified.drop('Class', axis=1).values
y = df_stratified[['Class']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular a matriz de covariância inversa para Mahalanobis
cov_matrix = np.cov(X_train.T)
cov_inv = np.linalg.inv(cov_matrix)

# Função para o KNN
def knn(X_train, y_train, X_test, k, dist_func, cov_inv=None):
    y_pred = []

    for test_point in X_test:
        # Calculando distâncias do ponto de teste para todos os pontos de treino
        distances = []
        for train_point, train_label in zip(X_train, y_train):
            # Passar cov_inv apenas se a função de distância for Mahalanobis
            if dist_func == dist_mahalanobis:
                dist = dist_func(test_point, train_point, cov_inv)  # Passando cov_inv para Mahalanobis
            else:
                dist = dist_func(test_point, train_point)
            distances.append((dist, train_label[0]))

        distances.sort(key=lambda x: x[0])
        k_neighbors = [label for _, label in distances[:k]]
        majority_class = max(set(k_neighbors), key=k_neighbors.count)
        y_pred.append(majority_class)

    return np.array(y_pred)

# Calcular a acurácia
k = 7
lista_dist = [dist_euclidiana, dist_manhattan, dist_mahalanobis, dist_chebyshev]

for dist_func in lista_dist:
    # Com Mahalanobis, passamos a matriz de covariância inversa
    if dist_func == dist_mahalanobis:
        y_pred = knn(X_train, y_train, X_test, k, dist_func, cov_inv=cov_inv)
    else:
        y_pred = knn(X_train, y_train, X_test, k, dist_func)
    acertos = np.sum(y_test.flatten() == y_pred.flatten())
    porcentagem_acertos = 100 * acertos / len(y_test)

    print(f"Acurácia do modelo {dist_func.__name__}: {porcentagem_acertos:.2f}%")
