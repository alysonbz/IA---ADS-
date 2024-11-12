# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


wine = load_wine_dataset()

# Inicializer o scale
scaler = StandardScaler()

# exclua do dataset a coluna Quality
X = wine.drop(['Quality'],axis=1)

#normalize o dataset com scaler
X_norm = scaler.fit_transform(X)
X_log = scaler.fit_transform(np.log(X))

#obtenha as labels da coluna Quality
y = wine['Quality'].values

#print a valriância de X
print('variancia', np.var(X))

#print a variânca do dataset X_norm
print('variancia do dataset normalizado', X_norm.var())

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)
X_train_log, X_test_log, y_train, y_test = train_test_split(X_log, y, stratify=y, random_state=42)

#inicialize o algoritmo KNN
knn = KNeighborsClassifier()

# Aplique a função fit do KNN
knn.fit(X_train,y_train)

# Verifique o acerto do classificador
print('score padrão', knn.score(X_test, y_test))

# Aplique a função fit do KNN
knn.fit(X_train_norm,y_train)

# Verifique o acerto do classificador
print('score normalizado', knn.score(X_test_norm, y_test))

# Aplique a função fit do KNN
knn.fit(X_train_log,y_train)

# Verifique o acerto do classificador
print('score log', knn.score(X_test_log, y_test))