import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Carregar o dataset
dataset = pd.read_csv("Cancer_Data_Adjusted.csv")

# Dividir em características e rótulos
X = dataset.drop("diagnosis", axis=1).values
y = dataset["diagnosis"].values

# Remover classes com menos de 2 instâncias (se necessário)
classes, counts = np.unique(y, return_counts=True)
valid_classes = classes[counts > 1]
X = X[np.isin(y, valid_classes)]
y = y[np.isin(y, valid_classes)]

# Separar conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)

# Normalização logarítmica
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# Normalização de média zero e variância unitária
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Função para avaliar o KNN
def evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=7, metric="manhattan"):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Avaliar normalizações
accuracy_log = evaluate_knn(X_train_log, X_test_log, y_train, y_test)
accuracy_norm = evaluate_knn(X_train_norm, X_test_norm, y_train, y_test)

# Resultados
print(f"Acurácia com normalização logarítmica: {accuracy_log * 100:.2f}%")
print(f"Acurácia com média zero e variância unitária: {accuracy_norm * 100:.2f}%")
