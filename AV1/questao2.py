import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
from collections import Counter
import os

# Caminho do arquivo de entrada
file_path = os.path.join(os.path.dirname(__file__), 'drug200_ajustado.csv')

# 1. Carregar o dataset ajustado
dataset = pd.read_csv(file_path)

# 2. Separar features (X) e alvo (y)
X = dataset.drop(columns=['Drug'])  # Recursos
y = dataset['Drug']  # Alvo

# Codificar colunas categóricas com get_dummies
X_encoded = pd.get_dummies(X, drop_first=True)

# 3. Garantir que todas as colunas de X sejam numéricas
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')

# 4. Verificar a forma de X_train
print("Forma de X_train:", X_encoded.shape)

# 5. Verificar se X_encoded tem mais de uma coluna
if X_encoded.shape[1] < 2:
    raise ValueError("X_encoded precisa ter pelo menos duas colunas para calcular a covariância.")

# 6. Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 7. Verificar novamente os dados de X_train
print("Primeiras linhas de X_train:\n", X_train.head())

# 8. Garantir que X_train tem apenas colunas numéricas
# Convertendo valores booleanos em 0 e 1 para garantir consistência
X_train = X_train.astype(float)

# 9. Calcular a matriz de covariância invertida para Mahalanobis
cov_matrix = np.cov(X_train.T)  # Transposta para garantir que as variáveis são as colunas
VI = np.linalg.inv(cov_matrix)

# 10. Implementar o KNN
def knn(X_train, y_train, X_test, k, distance_func):
    y_pred = []
    for test_point in X_test.values:
        distances = []
        for i, train_point in enumerate(X_train.values):
            dist = distance_func(test_point, train_point)
            distances.append((dist, y_train.iloc[i]))
        # Ordenar pelo menor distância e pegar os k mais próximos
        k_neighbors = sorted(distances, key=lambda x: x[0])[:k]
        # Votação majoritária
        classes = [neighbor[1] for neighbor in k_neighbors]
        majority_vote = Counter(classes).most_common(1)[0][0]
        y_pred.append(majority_vote)
    return y_pred

# 11. Funções de distância
def mahalanobis_dist(x, y, VI=VI):
    return mahalanobis(x, y, VI)

def chebyshev_dist(x, y):
    return chebyshev(x, y)

def manhattan_dist(x, y):
    return cityblock(x, y)

def euclidean_dist(x, y):
    return euclidean(x, y)

# 12. Avaliar acurácia para diferentes distâncias
k = 7
distance_functions = {
    "Mahalanobis": mahalanobis_dist,
    "Chebyshev": chebyshev_dist,
    "Manhattan": manhattan_dist,
    "Euclidean": euclidean_dist,
}

results = {}
for name, func in distance_functions.items():
    try:
        y_pred = knn(X_train, y_train, X_test, k, func)
        accuracy = np.mean(y_pred == y_test)
        results[name] = accuracy
    except Exception as e:
        results[name] = f"Erro: {e}"

# Exibir resultados
print("Resultados de Acurácia:")
for distance, accuracy in results.items():
    print(f"{distance}: {accuracy}")
