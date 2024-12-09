import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Funções das distâncias
def dist_euclidiana(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))


def dist_manhattan(v1, v2):
    return sum(abs(v1[i] - v2[i]) for i in range(len(v1)))


def dist_chebyshev(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1)))


def dist_mahalanobis(v1, v2, VI):
    return mahalanobis(v1, v2, VI)


def calcular_distancia(v1, v2, tipo="euclidiana", VI=None):
    if tipo == "euclidiana":
        return dist_euclidiana(v1, v2)
    elif tipo == "manhattan":
        return dist_manhattan(v1, v2)
    elif tipo == "chebyshev":
        return dist_chebyshev(v1, v2)
    elif tipo == "mahalanobis":
        if VI is None:
            raise ValueError("Matriz de covariância inversa é necessária para Mahalanobis.")
        return dist_mahalanobis(v1, v2, VI)
    else:
        raise ValueError("Tipo de distância não suportado.")


def knn(treinamento, nova_amostra, K, tipo_distancia="euclidiana", VI=None):
    dists = {}
    for i in range(len(treinamento)):
        d = calcular_distancia(treinamento[i][:-1], nova_amostra, tipo=tipo_distancia, VI=VI)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]
    classes = [treinamento[i][-1] for i in k_vizinhos]
    return max(set(classes), key=classes.count)


dataframe = pd.read_csv('datasets/train_ajustado.csv')
X = dataframe.drop(['price_range'], axis=1).values
y = dataframe['price_range'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

treinamento = np.column_stack((X_train, y_train))
teste = np.column_stack((X_test, y_test))

# Matriz inversa de covariância para Mahalanobis
VI = np.linalg.inv(np.cov(X_train, rowvar=False))

k = 7
resultados = {}

for tipo in ["euclidiana", "manhattan", "chebyshev", "mahalanobis"]:
    y_pred = []
    for amostra in teste:
        classe_pred = knn(treinamento, amostra[:-1], k, tipo_distancia=tipo, VI=VI)
        y_pred.append(classe_pred)

    print(f"\nRelatório de classificação com distância {tipo}:")
    print(classification_report(y_test, y_pred))

    acertos = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i])
    acuracia = 100 * acertos / len(teste)
    resultados[tipo] = acuracia

print("\nResumo das Acurácias:")
for tipo, acuracia in resultados.items():
    print(f"{tipo.capitalize()}: {acuracia:.2f}%")
