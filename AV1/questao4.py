#1 importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar o dataset
Cancer_Data = pd.read_csv('Cancer_Data_Adjusted.csv')

# Separar as colunas de features e target
X = Cancer_Data.drop(columns=["diagnosis"]).values
y = Cancer_Data["diagnosis"].values

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização
scaler = StandardScaler()

# Normalização de média zero e variância unitária
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Testar diferentes valores de k (número de vizinhos)
neighbors = np.arange(1, 21)
train_accuracies = {}
test_accuracies = {}

# Loop para testar diferentes valores de k
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train_norm, y_train)

    train_accuracies[neighbor] = knn.score(X_train_norm, y_train)
    test_accuracies[neighbor] = knn.score(X_test_norm, y_test)

# Mostrar as acurácias
print("Acurácia no treino: ", train_accuracies)
print("Acurácia no teste: ", test_accuracies)

# Plotando o gráfico de comparação das acurácias
plt.title("KNN: Variação do Número de Vizinhos (k)")

# Plotar as acurácias de treino
plt.plot(neighbors, train_accuracies.values(), label="Acurácia no Treinamento", color='blue')

# Plotar as acurácias de teste
plt.plot(neighbors, test_accuracies.values(), label="Acurácia no Teste", color='orange')

# Melhor valor de k baseado na acurácia de teste
best_k = max(test_accuracies, key=test_accuracies.get)
plt.axvline(x=best_k, linestyle='--', color='red', label=f"Melhor k = {best_k}")

# Adicionar legendas e rótulos
plt.legend()
plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("Acurácia")
plt.grid(True)
plt.tight_layout()

# Exibir o gráfico
plt.show()

# Imprimir o melhor k e a acurácia associada
print(f"O melhor valor de k é {best_k} com uma acurácia de {test_accuracies[best_k] * 100:.2f}% no conjunto de teste.")
