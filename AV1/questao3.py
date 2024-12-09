import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Carregar o dataset
WineQT = pd.read_csv('./dataset/wineqt_ajustado.csv')

# Separando as variáveis independentes (X) e a variável alvo (y)
X = WineQT.drop("quality", axis=1).values
y = WineQT['quality'].values

# Dividindo os dados em conjuntos de treino e teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Aplicando transformação logarítmica
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# Aplicando normalização com média zero e variância unitária
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Função para calcular a acurácia utilizando o KNN com métrica de Mahalanobis
def knn_mahalanobis(X_train, X_test, y_train, y_test, n_neighbors=10):
    # Calculando a matriz de covariância e sua inversa
    cov_matrix = np.cov(X_train, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Configurando o KNN com a métrica de Mahalanobis
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='mahalanobis', metric_params={'VI': inv_cov_matrix})

    # Treinando e testando o modelo
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Avaliando a acurácia com os dados transformados
accuracy_log = knn_mahalanobis(X_train_log, X_test_log, y_train, y_test)
accuracy_norm = knn_mahalanobis(X_train_norm, X_test_norm, y_train, y_test)

# Exibindo os resultados
print(f"Acurácia com normalização logarítmica: {accuracy_log * 100:.2f}%")
print(f"Acurácia com média zero e variância unitária: {accuracy_norm * 100:.2f}%")
