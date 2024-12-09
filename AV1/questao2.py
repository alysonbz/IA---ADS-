import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
import numpy as np

# 1. Importar o dataset atualizado
data = pd.read_csv(r'C:\Users\pinheiroiwnl\Desktop\AV1\IA---ADS-\AV1\diabestes_ajustado.csv')

# 2. Dividir o dataset em treino e teste
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Função para calcular a acurácia considerando diferentes métricas de distância
def evaluate_knn_with_distance(X_train, X_test, y_train, y_test, metric, metric_name):
    knn = KNeighborsClassifier(n_neighbors=7, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia com a distância {metric_name}: {accuracy:.4f}")
    return accuracy

# 4. Avaliar o KNN com diferentes métricas de distância
print("\nResultados do KNN com diferentes distâncias:")

# a) Distância de Mahalanobis
# Calcular a matriz de covariância para Mahalanobis
data_cov = np.cov(X_train, rowvar=False)
data_cov_inv = np.linalg.inv(data_cov)
def mahalanobis_metric(u, v):
    return mahalanobis(u, v, data_cov_inv)
evaluate_knn_with_distance(X_train, X_test, y_train, y_test, mahalanobis_metric, "Mahalanobis")

# b) Distância de Chebyshev
evaluate_knn_with_distance(X_train, X_test, y_train, y_test, "chebyshev", "Chebyshev")

# c) Distância de Manhattan
evaluate_knn_with_distance(X_train, X_test, y_train, y_test, "manhattan", "Manhattan")

# d) Distância Euclidiana
evaluate_knn_with_distance(X_train, X_test, y_train, y_test, "euclidean", "Euclidiana")