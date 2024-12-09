
import math

import pandas as pd


def dist_euclidiana(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1) - 1)))

def dist_manhattan(v1, v2):
    return sum(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))

def dist_minkowski(v1, v2, p=3):
    return sum(abs(v1[i] - v2[i]) ** p for i in range(len(v1) - 1)) ** (1 / p)

def dist_chebyshev(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))

def calcular_distancia(v1, v2, tipo="euclidiana", p=3):
    if tipo == "euclidiana":
        return dist_euclidiana(v1, v2)
    elif tipo == "manhattan":
        return dist_manhattan(v1, v2)
    elif tipo == "minkowski":
        return dist_minkowski(v1, v2, p)
    elif tipo == "chebyshev":
        return dist_chebyshev(v1, v2)
    else:
        raise ValueError("Tipo de distância não suportado.")


def knn(treinamento, nova_amostra, K, tipo_distancia="euclidiana", p=3):
    dists = {}
    for i in range(len(treinamento)):
        d = calcular_distancia(treinamento[i], nova_amostra, tipo=tipo_distancia, p=p)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]
    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0

    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0



lista = []
with open('../dataset/iris.data', 'r') as f:
    for linha in f.readlines():
        a = linha.replace('\n', '').split(',')
        if a[4] == 'Iris-setosa':
            a[4] = 1.0
        elif a[4] == 'Iris-versicolor':
            a[4] = 2.0
        elif a[4] == 'Iris-virginica':
            a[4] = 3.0

        valor = [float(n) for n in a]
        lista.append(valor)

pd.DataFrame(lista).info()

def countclasses(lista):
    setosa, versicolor, virginica = 0, 0, 0
    for i in range(len(lista)):
        if lista[i][4] == 1.0:
            setosa += 1
        if lista[i][4] == 2.0:
            versicolor += 1
        if lista[i][4] == 3.0:
            virginica += 1
    return [setosa, versicolor, virginica]

p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
total1, total2, total3 = 0, 0, 0

for lis in lista:
    if lis[-1] == 1.0 and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[-1] == 2.0 and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[-1] == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)

acertos, K = 0, 1
tipo_distancia = "euclidiana"  # "manhattan", "minkowski", "chebyshev"
p_minkowski = 3  # Valor de 'p' para Minkowski

for amostra in teste:
    classe = knn(treinamento, amostra, K, tipo_distancia=tipo_distancia, p=p_minkowski)
    if amostra[-1] == classe:
        acertos += 1

print(f"Porcentagem de acertos usando {tipo_distancia}: {100 * acertos / len(teste):.2f}%")
