# IMPLEMENTAÇÃO MANUAL DO KNN E OUTROS ALGORITMOS
import math
import pandas as pd

data_path = '../dataset/iris/iris.data'
lista = []
with open(data_path, 'r') as f:
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
    setosa = 0
    versicolor = 0
    virginica = 0
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
total1 = 0
total2 = 0
total3 = 0
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


def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)

    return math.sqrt(soma)


def dist_manhattan(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):  # Exclude the class label
        soma += abs(v1[i] - v2[i])
    return soma


def dist_minkowski(v1, v2, p):
    dim, soma = len(v1), 0
    for i in range(dim - 1):  # Exclude class label
        soma += math.pow(abs(v1[i] - v2[i]), p)
    return math.pow(soma, 1 / p)


def dist_chebyshev(v1, v2):
    dim, max_dif = len(v1), 0
    for i in range(dim - 1):  # Exclude class label
        max_diff = max(max_dif, abs(v1[i] - v2[i]))
    return max_dif


def knn(treinamento, nova_amostra, K, dist_func):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_func(treinamento[i], nova_amostra)  # Use the selected distance function
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


print("Distância Euclidiana:")
acertos = 0
for amostra in teste:
    classe = knn(treinamento, amostra, K=3, dist_func=dist_euclidiana)
    if amostra[-1] == classe:
        acertos += 1
print("Porcentagem de acertos:", 100 * acertos / len(teste))

print("Distância de Manhattan:")
acertos = 0
for amostra in teste:
    classe = knn(treinamento, amostra, K=3, dist_func=dist_manhattan)
    if amostra[-1] == classe:
        acertos += 1
print("Porcentagem de acertos:", 100 * acertos / len(teste))

print("Distância de Minkowski com p=3:")
acertos = 0
for amostra in teste:
    classe = knn(treinamento, amostra, K=3, dist_func=lambda v1, v2: dist_minkowski(v1, v2, p=3))
    if amostra[-1] == classe:
        acertos += 1
print("Porcentagem de acertos:", 100 * acertos / len(teste))

print("Distância de Chebyshev:")
acertos = 0
for amostra in teste:
    classe = knn(treinamento, amostra, K=3, dist_func=dist_chebyshev)
    if amostra[-1] == classe:
        acertos += 1
print("Porcentagem de acertos:", 100 * acertos / len(teste))
