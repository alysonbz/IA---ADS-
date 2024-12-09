import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Importar o dataset atualizado

data = pd.read_csv(r'C:\Users\pinheiroiwnl\Desktop\AV1\IA---ADS-\AV1\diabestes_ajustado.csv')

# 2. Dividir o dataset em treino e teste
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalizar o conjunto de dados com normalização logarítmica
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# 4. Avaliar o KNN com dados log-normalizados (melhor distância: Manhattan)
knn_log = KNeighborsClassifier(n_neighbors=7, metric="manhattan")
knn_log.fit(X_train_log, y_train)
y_pred_log = knn_log.predict(X_test_log)
accuracy_log = accuracy_score(y_test, y_pred_log)

# 5. Normalizar o conjunto de dados com média zero e variância unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Avaliar o KNN com dados escalados (média zero e variância unitária)
knn_scaled = KNeighborsClassifier(n_neighbors=7, metric="manhattan")
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

# 7. Comparar as acurácias
print(f"Acurácia com normalização logarítmica: {accuracy_log:.4f}")
print(f"Acurácia com média zero e variância unitária: {accuracy_scaled:.4f}")