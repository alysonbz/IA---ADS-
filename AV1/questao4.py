import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

dataframe = pd.read_csv('datasets/train_ajustado.csv')
X = dataframe.drop(['price_range'], axis=1).values
y = dataframe['price_range'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

k_values = range(1, 21)
acuracias = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acuracias.append(accuracy_score(y_test, y_pred))

melhor_k = k_values[np.argmax(acuracias)]
melhor_acuracia = max(acuracias)

plt.figure(figsize=(10, 6))
plt.plot(k_values, acuracias, marker='o', linestyle='-', color='b')
plt.title("Acurácia do KNN para diferentes valores de k (distância Manhattan)")
plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("Acurácia")
plt.xticks(k_values)
plt.grid(True)
plt.axvline(x=melhor_k, color='r', linestyle='--', label=f'Melhor k = {melhor_k}')
plt.legend()
plt.show()

print(f"Melhor acurácia: {melhor_acuracia * 100:.2f}%")
