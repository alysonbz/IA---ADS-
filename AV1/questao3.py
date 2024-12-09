import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os

# Caminho do arquivo ajustado
file_path = os.path.join(os.path.dirname(__file__), 'drug200_ajustado.csv')

# 1. Carregar o dataset ajustado
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

# Verificar se X_encoded tem mais de uma coluna
if X_encoded.shape[1] < 2:
    raise ValueError("X_encoded precisa ter pelo menos duas colunas para calcular a covariância.")

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Garantir que X_train seja uma matriz numpy 2D antes de calcular a covariância
X_train_np = X_train.to_numpy()

# Função de cálculo da distância de Mahalanobis
cov_matrix = np.cov(X_train_np.T)  # Transposta para garantir que as variáveis são as colunas
VI = np.linalg.inv(cov_matrix)

def mahalanobis_dist(x, y, VI=VI):
    # Garantir que x e y sejam vetores 1-D e converter para numpy array
    x = np.array(x).flatten().astype(np.float64)
    y = np.array(y).flatten().astype(np.float64)
    return mahalanobis(x, y, VI)

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

# 4. Normalização logarítmica
X_train_log = X_train.apply(np.log1p)
X_test_log = X_test.apply(np.log1p)

# Garantir que X_train_log e X_test_log sejam numéricos após log
X_train_log = X_train_log.apply(pd.to_numeric, errors='coerce')
X_test_log = X_test_log.apply(pd.to_numeric, errors='coerce')

# Substituir quaisquer NaN ou inf por 0 ou a média das colunas
X_train_log = X_train_log.replace([np.inf, -np.inf], 0).fillna(0)
X_test_log = X_test_log.replace([np.inf, -np.inf], 0).fillna(0)

# Garantir que os dados sejam float64 após a transformação logarítmica
X_train_log = X_train_log.astype(np.float64)
X_test_log = X_test_log.astype(np.float64)

# Verificar os tipos de dados após conversão
print(f"Tipos de dados após normalização logarítmica: \n{X_train_log.dtypes}")

# Verificar valores de X_train_log e X_test_log para garantir que estão numéricos
print(f"Valores de X_train_log: \n{X_train_log.head()}")
print(f"Valores de X_test_log: \n{X_test_log.head()}")

# Verificar se há valores não numéricos antes de passar para o cálculo da distância
assert np.issubdtype(X_train_log.dtypes.iloc[0], np.floating), "X_train_log contém valores não numéricos!"
assert np.issubdtype(X_test_log.dtypes.iloc[0], np.floating), "X_test_log contém valores não numéricos!"

# Verificar a presença de NaN ou inf em X_train_log e X_test_log
assert not X_train_log.isna().any().any(), "X_train_log contém NaN!"
assert not X_test_log.isna().any().any(), "X_test_log contém NaN!"
assert not (np.isinf(X_train_log).any().any()), "X_train_log contém inf!"
assert not (np.isinf(X_test_log).any().any()), "X_test_log contém inf!"

# Acurácia do KNN com normalização logarítmica
y_pred_log = knn(X_train_log.to_numpy(), y_train, X_test_log.to_numpy(), k=7, distance_func=mahalanobis_dist)
accuracy_log = np.mean(y_pred_log == y_test)

# 5. Normalização de média zero e variância unitária (padronização)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Acurácia do KNN com normalização de média zero e variância unitária
y_pred_scaled = knn(X_train_scaled, y_train, X_test_scaled, k=7, distance_func=mahalanobis_dist)
accuracy_scaled = np.mean(y_pred_scaled == y_test)

# 6. Comparação das acurácias
print(f"Acurácia com normalização logarítmica: {accuracy_log}")
print(f"Acurácia com normalização de média zero e variância unitária: {accuracy_scaled}")
