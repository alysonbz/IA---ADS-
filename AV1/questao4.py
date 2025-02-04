import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Carregar o dataset
WineQT = pd.read_csv('./dataset/wineqt_ajustado.csv')

# Dividir os dados em features (X) e labels (y)
X = WineQT.drop("quality", axis=1).values
y = WineQT['quality'].values

# Normalizar os dados (média zero e variância unitária)
X_train_log = np.log1p(X)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Intervalo de vizinhos a ser testado
neighbors_range = np.arange(1, 26)
train_accuracies = []
test_accuracies = []

# Avaliar o desempenho para cada número de vizinhos
for n_neighbors in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))

# Exibir os resultados
print("Acurácias no treino:", train_accuracies)
print("Acurácias no teste:", test_accuracies)

# Plotar os resultados
plt.figure(figsize=(8, 6))
plt.plot(neighbors_range, train_accuracies, label="Acurácia no Treino", marker='o')
plt.plot(neighbors_range, test_accuracies, label="Acurácia no Teste", marker='o')
plt.title("KNN: Variando o Número de Vizinhos")
plt.xlabel("Número de Vizinhos")
plt.ylabel("Acurácia")
plt.legend()
plt.grid()
plt.show()
