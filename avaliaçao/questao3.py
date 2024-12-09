import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis, cityblock, euclidean, chebyshev
from sklearn.preprocessing import StandardScaler


#unção KNN
def knn_predict(X_train, y_train, X_test, k, distance_metric="euclidean"):
    predictions = []
    cov_matrix = np.cov(X_train.T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

    for test_point in X_test:
        distances = []
        for train_point, label in zip(X_train, y_train):
            if distance_metric == "mahalanobis":
                dist = mahalanobis(test_point, train_point, inv_cov_matrix)
            elif distance_metric == "chebyshev":
                dist = np.max(np.abs(test_point - train_point))
            elif distance_metric == "manhattan":
                dist = np.sum(np.abs(test_point - train_point))
            elif distance_metric == "euclidean":
                dist = np.sqrt(np.sum((test_point - train_point) ** 2))
            else:
                raise ValueError("Métrica de distância inválida.")
            distances.append((dist, label))


        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]
        predictions.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
    return np.array(predictions)

file_path = "healtcare_ajustado.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=["stroke"])
y = data["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#normalização logarítmica
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

#normalização de média zero e variância unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 7
best_metric = "chebyshev"  #baseado na questao anterior

y_pred_log = knn_predict(X_train_log.to_numpy(), y_train.to_numpy(), X_test_log.to_numpy(), k, best_metric)
accuracy_log = accuracy_score(y_test, y_pred_log)

# Avaliar com normalização de média zero e variância unitária
y_pred_scaled = knn_predict(X_train_scaled, y_train.to_numpy(), X_test_scaled, k, best_metric)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print(f"Acurácia com normalização logarítmica: {accuracy_log:.4f}")
print(f"Acurácia com normalização de média zero e variância unitária: {accuracy_scaled:.4f}")
