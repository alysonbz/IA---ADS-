import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

dataframe = pd.read_csv('datasets/train_ajustado.csv')
X = dataframe.drop(['price_range'], axis=1).values
y = dataframe['price_range'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7, metric="manhattan")

# Avaliação sem normalização
knn.fit(X_train, y_train)
y_pred_sem_normalizacao = knn.predict(X_test)
acuracia_sem_normalizacao = accuracy_score(y_test, y_pred_sem_normalizacao)

# Normalização Logarítmica
X_train_log = np.log1p(X_train)  # log(1 + x)
X_test_log = np.log1p(X_test)

knn.fit(X_train_log, y_train)
y_pred_log = knn.predict(X_test_log)
acuracia_log = accuracy_score(y_test, y_pred_log)

# Normalização de Média Zero e Variância Unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)
acuracia_scaled = accuracy_score(y_test, y_pred_scaled)

print("Comparação de Acurácias:")
print(f"Sem Normalização: {acuracia_sem_normalizacao * 100:.2f}%")
print(f"Com Normalização Logarítmica: {acuracia_log * 100:.2f}%")
print(f"Com Normalização (Média Zero e Variância Unitária): {acuracia_scaled * 100:.2f}%")
