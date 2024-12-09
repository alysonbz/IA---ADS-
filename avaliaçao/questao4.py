import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Função KNN (copiada/adaptada da Questão 2)
def knn_predict(X_train, y_train, X_test, k, distance_metric):
    predictions = []
    for test_point in X_test:
        distances = []
        for train_point, label in zip(X_train, y_train):
            if distance_metric == "euclidean":
                dist = np.sqrt(np.sum((test_point - train_point) ** 2))
            elif distance_metric == "manhattan":
                dist = np.sum(np.abs(test_point - train_point))
            elif distance_metric == "chebyshev":
                dist = np.max(np.abs(test_point - train_point))
            else:
                raise ValueError("Métrica de distância inválida.")
            distances.append((dist, label))

        #denar por distância e pegar os k vizinhos mais próximos
        distances = sorted(distances, key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]

        predictions.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
    return predictions


data = pd.read_csv("healtcare_ajustado.csv")

X = data.drop("stroke", axis=1)
y = data["stroke"]

X_normalized = np.log1p(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

X_train_array = X_train.to_numpy()
X_test_array = X_test.to_numpy()

# Testar diferentes valores de k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    y_pred = knn_predict(X_train_array, y_train.to_numpy(), X_test_array, k, distance_metric="euclidean")
    accuracy = np.mean(y_pred == y_test.to_numpy())
    accuracies.append(accuracy)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title("Acurácia do KNN para diferentes valores de k")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia")
plt.xticks(k_values)
plt.grid()
plt.show()

# Melhor valor de k
best_k = k_values[np.argmax(accuracies)]
print(f"O melhor valor de k é: {best_k} com uma acurácia de {max(accuracies):.2f}")
