import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import matplotlib.pyplot as plt

from questao3 import knn_predict

data = pd.read_csv('healtcare_ajustado.csv')
X = data.iloc[:, :-1].values  # Todas as colunas, exceto a última
y = data.iloc[:, -1].values   # Apenas a última coluna

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def knn(X_train, y_train, X_test, k, distance_func):
    y_pred = []
    for x_test in X_test:
        distances = [distance_func(x_test, x_train) for x_train in X_train]
        k_neighbors = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_neighbors]
        y_pred.append(np.bincount(k_labels).argmax())
    return np.array(y_pred)

def chebyshev_distance(x, y):
    return distance.chebyshev(x, y)

def manhattan_distance(x, y):
    return distance.cityblock(x, y)

def euclidean_distance(x, y):
    return distance.euclidean(x, y)

k = 7
accuracy = {}

distance_functions = {
    "Chebyshev": chebyshev_distance,
    "Manhattan": manhattan_distance,
    "Euclidean": euclidean_distance
}

for name, func in distance_functions.items():
    y_pred = knn(X_train, y_train, X_test, k, func)
    accuracy[name] = np.mean(y_pred == y_test)

print("Acurácia para diferentes funções de distância:")
for name, acc in accuracy.items():
    print(f"{name}: {acc:.2f}")


metrics = ["chebyshev", "manhattan", "euclidean"]
results = {}

for metric in metrics:
    print(f"Testando com a métrica: {metric}")
    y_pred = knn_predict(X_train_array, y_train, X_test_array, k=7, distance_metric=metric)
    accuracy = accuracy_score(y_test, y_pred)
    results[metric] = accuracy
    print(f"Acurácia com {metric}: {accuracy:.4f}")

# Encontrar a melhor métrica
best_metric = max(results, key=results.get)
print(f"\nMelhor métrica de distância: {best_metric}")
print(f"Acurácia da melhor métrica: {results[best_metric]:.4f}")

