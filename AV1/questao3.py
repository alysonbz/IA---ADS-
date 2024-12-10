import numpy as np
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

# Função para normalização logarítmica com proteção contra zeros ou negativos
def normalizar_log(X):
    # Debugar aqui - valores nan (Substituir valores negativos e zero por um valor pequeno)
    X = np.where(X <= 0, 1e-10, X)
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
k = 7
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

# Avaliar a acurácia com normalização logarítmica
knn.fit(X_train_log, y_train)
acuracia_log = knn.score(X_test_log, y_test)

# Avaliar a acurácia com normalização de média zero e variância unitária
knn.fit(X_train_std, y_train)
acuracia_std = knn.score(X_test_std, y_test)

# Comparar os resultados
print(f"Acurácia com normalização logarítmica: {acuracia_log * 100:.2f}%")
print(f"Acurácia com normalização de média zero e variância unitária: {acuracia_std * 100:.2f}%")

