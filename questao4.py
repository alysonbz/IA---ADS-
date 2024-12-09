import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Importar o dataset atualizado
data = pd.read_csv(r'C:\Users\pinheiroiwnl\Downloads\prova_GUI\prova_GUI\diabestes_ajustado.csv')

# 2. Dividir o dataset em treino e teste
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Aplicar a melhor normalização (média zero e variância unitária, com base no exercício anterior)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Encontrar o melhor valor de k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# 5. Plotar o gráfico das acurácias
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker="o")
plt.title("Acurácia do KNN para diferentes valores de k")
plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("Acurácia")
plt.xticks(k_values)
plt.grid()
plt.show()

# 6. Indicar o melhor k
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"Melhor k: {best_k} com acurácia de {best_accuracy:.4f}")