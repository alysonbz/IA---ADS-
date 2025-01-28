# Com base nas paramentrizações vistas anteriormento, neste exercicio você deve buscar saber a melhor parametrização
# do knn implementado na questão anterior. (lembrete: O Dataset nao esta sendo encontrado, ver utils.py)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.utils import Hotel_Normalizado

Hotel = Hotel_Normalizado()
print("Dimensão do dataset:", Hotel)

# Separar o dataset em X (features) e y (classe/target)
X = Hotel.drop(columns=["booking_status_Not_Canceled"]).values
y = Hotel["booking_status_Not_Canceled"].values

# Divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função para normalização logarítmica
def normalizar_log(X):
    transformer = FunctionTransformer(np.log1p, validate=True)  # Log transform
    return transformer.fit_transform(X)

# Normalização Logarítmica
X_train_log = normalizar_log(X_train)
X_test_log = normalizar_log(X_test)

# Normalização de Média Zero e Variância Unitária
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Definir o modelo KNN com k=7
k = 61
knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

# Função para calcular a acurácia para diferentes valores de k
def avaliar_k(knn, X_train, y_train, X_test, y_test, max_k=20):
    acuracias = []
    for k in range(1, max_k + 1):
        knn.set_params(n_neighbors=7)  # Ajusta o valor de k pra 7(testar)
        knn.fit(X_train, y_train)
        acuracia = knn.score(X_test, y_test)
        acuracias.append(acuracia)
    return acuracias

# Avaliar a acurácia para diferentes valores de k com normalização logarítmica
acuracias_log = avaliar_k(knn, X_train_log, y_train, X_test_log, y_test)

# Avaliar a acurácia para diferentes valores de k - com normalização de média zero e variância unitária
acuracias_std = avaliar_k(knn, X_train_std, y_train, X_test_std, y_test)

# Plotando o gráfico para comparar as acurácias para os diferentes valores de k
plt.figure(figsize=(10,6))
plt.plot(range(1, 21), acuracias_log, label='Logarítmica', color='blue', marker='o')
plt.plot(range(1, 21), acuracias_std, label='Média Zero e Variância Unitária', color='red', marker='x')
plt.title('Acurácia do KNN para diferentes valores de k')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir o melhor valor de k para cada tipo de normalização
melhor_k_log = np.argmax(acuracias_log) + 1
melhor_k_std = np.argmax(acuracias_std) + 1

print(f"Melhor valor de k para normalização logarítmica: {melhor_k_log}")
print(f"Melhor valor de k para normalização de média zero e variância unitária: {melhor_k_std}")