from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from AV1.src.utils import load_water_quality
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

waterQuality = load_water_quality()

X = waterQuality.drop(['is_safe'], axis=1).values
y = waterQuality['is_safe'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cov_matrix = np.cov(X_train_scaled.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)

def evaluate_knn_mahalanobis(X_train, X_test, y_train, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='mahalanobis',
                               metric_params={'VI': inv_cov_matrix})
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

k_values = range(1, 21)
accuracies = []

for k in k_values:
    acc = evaluate_knn_mahalanobis(X_train_scaled, X_test_scaled, y_train, y_test, n_neighbors=k)
    accuracies.append(acc)

best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Acurácia')
plt.axvline(best_k, linestyle='--', color='r', label=f'Melhor K = {best_k}')
plt.title('Acurácia do KNN com Distância de Mahalanobis')
plt.xlabel('Número de Vizinhos (K)')
plt.ylabel('Acurácia')
plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()

print(f"Melhor K: {best_k}")
print(f"Acurácia com o melhor K: {best_accuracy * 100}%")
