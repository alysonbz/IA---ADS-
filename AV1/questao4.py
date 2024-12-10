import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

file_path = "../AV1/datasets/novo_creditcard.csv"
df = pd.read_csv(file_path)
class_samples = 30000
df_stratified = df.groupby('Class').sample(n=class_samples, random_state=42)

X = df_stratified.drop('Class', axis=1).values
y = df_stratified['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização Z-score
scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)
X_test_z = scaler.transform(X_test)

k_values = range(1, 51)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train_z, y_train)
    accuracy = knn.score(X_test_z, y_test)
    accuracies.append(accuracy)

melhor_k = k_values[np.argmax(accuracies)]
melhor_accuracy = max(accuracies)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Acurácia')
plt.axvline(x=melhor_k, color='r', linestyle='--', label=f'Melhor k = {melhor_k}')
plt.title('Acurácia do KNN para diferentes valores de k')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.legend()
plt.grid()
plt.show()

print(f"Melhor valor de k: {melhor_k}")
print(f"Melhor acurácia: {melhor_accuracy * 100:.2f}%")
