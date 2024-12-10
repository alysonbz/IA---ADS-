import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = "../AV1/datasets/novo_creditcard.csv"
df = pd.read_csv(file_path)

def dist_euclidiana(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

class_samples = 5000
df_stratified = df.groupby('Class').sample(n=class_samples, random_state=42)

X = df_stratified.drop('Class', axis=1).values
y = df_stratified[['Class']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização Logarítmica
X_train_pos = X_train - np.min(X_train) + 1
X_test_pos = X_test - np.min(X_test) + 1
X_train_log = np.log1p(X_train_pos)
X_test_log = np.log1p(X_test_pos)

# Normalização Z-score (média zero e variância unitária)
scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)
X_test_z = scaler.transform(X_test)


def knn(X_train, y_train, X_test, k):
    y_pred = []

    for test_point in X_test:
        distances = []
        for train_point, train_label in zip(X_train, y_train):
            dist = dist_euclidiana(test_point, train_point)
            distances.append((dist, train_label[0]))

        distances.sort(key=lambda x: x[0])
        k_neighbors = [label for _, label in distances[:k]]
        majority_class = max(set(k_neighbors), key=k_neighbors.count)
        y_pred.append(majority_class)

    return np.array(y_pred)


k = 7

y_pred_log = knn(X_train_log, y_train, X_test_log, k)
acertos_log = np.sum(y_test.flatten() == y_pred_log.flatten())
porcentagem_acertos_log = 100 * acertos_log / len(y_test)

y_pred_z = knn(X_train_z, y_train, X_test_z, k)
acertos_z = np.sum(y_test.flatten() == y_pred_z.flatten())
porcentagem_acertos_z = 100 * acertos_z / len(y_test)

print("Acurácia do KNN com diferentes normalizações:")
print(f"Logarítmica: {porcentagem_acertos_log:.2f}%")
print(f"Z-score: {porcentagem_acertos_z:.2f}%")
