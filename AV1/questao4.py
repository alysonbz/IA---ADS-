import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Carregar o dataset
WineQT = pd.read_csv('./dataset/wineqt_ajustado.csv')

# DataFrame com todas as colunas, com exceção de ``quality``
X = WineQT.drop("quality", axis=1).values

# Dataframe de labels com a coluna quality
y = WineQT['quality'].values

# Aplicando normalização media zero e variância unitária
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Separando o conjunto de teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

neighbors = np.arange(1, 26)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("acuracy on train: ",train_accuracies)
print("acuracy on test: ", test_accuracies)

plt.figure(figsize=(8,6))
plt.title("KNN: Varying Number of neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracies")
plt.plot(neighbors, test_accuracies.values(), label="Test Accuracies")
plt.legend()
plt.xlabel("Numero de neighbors")
plt.ylabel("Accuracies")
plt.show()