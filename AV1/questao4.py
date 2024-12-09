import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os

# 1. Obter o caminho do arquivo CSV
file_path = os.path.join(os.getcwd(), "drug200.csv")  # O arquivo CSV deve estar no mesmo diretório do script
dataset = pd.read_csv(file_path)

# 2. Separar as features (X) e alvo (y)
X = dataset.drop(columns=['Drug'])  # Recursos
y = dataset['Drug']  # Alvo

# Codificar colunas categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

# 3. Garantir que todas as colunas de X sejam numéricas e verificar valores faltantes
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')  # Forçar a conversão para numérico, erros para NaN

# Convertendo colunas booleanas para inteiros (True = 1, False = 0)
X_encoded = X_encoded.astype(int)

# Remover linhas com NaN, caso haja algum valor faltante
X_encoded = X_encoded.dropna()

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. Normalização de média zero e variância unitária (padronização)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Função de cálculo da distância Euclidiana
def euclidean_dist(x, y):
    return np.linalg.norm(x - y)

# Função KNN (adaptada para trabalhar com numpy.ndarray)
def knn(X_train, y_train, X_test, k, distance_func):
    y_pred = []
    for test_point in X_test:  # Alterado para acessar diretamente a linha do numpy.ndarray
        distances = []
        for i, train_point in enumerate(X_train):  # Alterado para acessar diretamente a linha do numpy.ndarray
            dist = distance_func(test_point, train_point)  # Passar vetores 1D
            distances.append((dist, y_train.iloc[i]))
        # Ordenar pelo menor distância e pegar os k mais próximos
        k_neighbors = sorted(distances, key=lambda x: x[0])[:k]
        # Votação majoritária
        classes = [neighbor[1] for neighbor in k_neighbors]
        majority_vote = Counter(classes).most_common(1)[0][0]
        y_pred.append(majority_vote)
    return y_pred

# 5. Testando diferentes valores de k e encontrando o melhor k
k_values = list(range(1, 11))  # Testar k de 1 a 10 para melhorar a performance
accuracies = []

for k in k_values:
    # Acurácia do KNN com a distância Euclidiana e o valor de k
    y_pred = knn(X_train_scaled, y_train, X_test_scaled, k=k, distance_func=euclidean_dist)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

# 6. Plotando o gráfico
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', color='b', label='Acurácia')
plt.title('Acurácia do KNN para diferentes valores de k')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.grid(True)
plt.xticks(k_values)
plt.legend()
plt.show()

# 7. Exibir o melhor k
best_k = k_values[np.argmax(accuracies)]
print(f'O melhor valor de k é: {best_k} com acurácia de {max(accuracies)}')
