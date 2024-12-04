from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import pandas as pd
import numpy as np
from AV1.questao1 import load_water_quality

waterQuality = load_water_quality()

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def chebyshev_distance(point1, point2):
    return np.max(np.abs(point1 - point2))

def mahalanobis_distance(point1, point2, cov_inv):
    diff = point1 - point2
    return np.sqrt(diff.T @ cov_inv @ diff)

def knn_manual(X_train, y_train, X_test, k=7, distance_func=euclidean_distance, cov_inv=None):
    predictions = []
    for test_point in X_test:
        # Calcular a distância entre o ponto de teste e todos os pontos de treinamento
        distances = [
            distance_func(test_point, train_point) if cov_inv is None
            else mahalanobis_distance(test_point, train_point, cov_inv)
            for train_point in X_train
        ]
        # Combinar as distâncias com os rótulos
        neighbors = sorted(zip(distances, y_train))[:k]
        # Extrair os rótulos dos K vizinhos mais próximos
        k_nearest_labels = [label for _, label in neighbors]
        # Determinar a classe predominante
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions

# Função para calcular a acurácia para diferentes distâncias
def evaluate_knn(X_train, y_train, X_test, y_test, k, distance_func, cov_inv=None):
    y_pred = knn_manual(X_train, y_train, X_test, k, distance_func, cov_inv)
    return accuracy_score(y_test, y_pred)

X = waterQuality[['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium',
                  'chloramine', 'chromium', 'copper', 'flouride', 'bacteria',
                  'viruses', 'lead', 'nitrates', 'nitrites', 'mercury',
                  'perchlorate', 'radium', 'selenium', 'silver', 'uranium']].values
y = waterQuality['is_safe'].values

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Avaliando diferentes métricas de distância
results = {}
for name, func in [
    ("Euclidiana", euclidean_distance),
    ("Manhattan", manhattan_distance),
    ("Chebyshev", chebyshev_distance),
]:
    acc = evaluate_knn(X_train, y_train, X_test, y_test, k=7, distance_func=func)
    results[name] = acc

# Avaliando a distância de Mahalanobis
cov_matrix = np.cov(X_train, rowvar=False)
cov_inv = np.linalg.inv(cov_matrix)
results["Mahalanobis"] = evaluate_knn(X_train, y_train, X_test, y_test, k=7, distance_func=mahalanobis_distance, cov_inv=cov_inv)

# Exibindo os resultados
for name, acc in results.items():
    print(f"Distância {name}: Acurácia = {acc}")

