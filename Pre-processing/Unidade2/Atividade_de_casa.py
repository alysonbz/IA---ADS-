import math

lista = []
with open('../dataset/iris.data', 'r') as f:
    for linha in f.readlines():
        a = linha.replace('\n', '').split(',')
        if a[4] == "Iris-setosa":
            a[4] = 1.0
        if a[4] == "Iris-versicolor":
            a[4] = 2.0
        if a[4] == "Iris-virginica":
            a[4] = 3.0

        valor = [float(num) for num in a]
        lista.append(valor)

print(lista)

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        #if len(lista[i]) > 3:
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
    for i in range(dim - 1):
        soma += abs(v1[i] - v2[i])
    return soma

def dist_minkowski(v1, v2, p=1):
    dim, soma = len(v1), 0
    for i in range(dim):
        soma += abs(v1[i] - v2[i]) ** p
    return soma ** (1 / p)

def dist_chebyshev(v1, v2):
    dim = len(v1)
    max_diferenca = 0
    for i in range(dim - 1):
        diferenca = abs(v1[i] - v2[i])
        if diferenca > max_diferenca:
            max_diferenca = diferenca
    return max_diferenca

def knn_eucli(treinamento_eu, nova_amostra, K):
    dists, len_treino = {}, len(treinamento_eu)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento_eu[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento_eu[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento_eu[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a_eu = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a_eu.index(max(a_eu)) + 1.0

def knn_manhattan(treinamento_man, nova_amostra, K):
    dists, len_treino = {}, len(treinamento_man)

    for i in range(len_treino):
        d = dist_manhattan(treinamento_man[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento_man[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento_man[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a_man = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a_man.index(max(a_man)) + 1.0

def knn_minkowski(treinamento_mink, nova_amostra, K):
    dists, len_treino = {}, len(treinamento_mink)

    for i in range(len_treino):
        d = dist_minkowski(treinamento_mink[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento_mink[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento_mink[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a_man = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a_man.index(max(a_man)) + 1.0

def knn_chebyshev(treinamento_che, nova_amostra, K):
    dists, len_treino = {}, len(treinamento_che)

    for i in range(len_treino):
        d = dist_chebyshev(treinamento_che[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento_che[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento_che[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a_man = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a_man.index(max(a_man)) + 1.0

acertos_eu, K_eu = 0, 1
acertos_man, K_man = 0, 1
acertos_mink, K_mink = 0, 1
acertos_che, K_che = 0, 1
for amostra_eu in teste:
    classe = knn_eucli(treinamento, amostra_eu, K_eu)
    if amostra_eu[-1] == classe:
        acertos_eu += 1
for amostra_man in teste:
    classe = knn_manhattan(treinamento, amostra_man, K_man)
    if amostra_man[-1] == classe:
        acertos_man += 1
for amostra_mink in teste:
    classe = knn_minkowski(treinamento, amostra_mink, K_mink)
    if amostra_mink[-1] == classe:
        acertos_mink += 1
for amostra_che in teste:
    classe = knn_chebyshev(treinamento, amostra_che, K_che)
    if amostra_che[-1] == classe:
        acertos_che += 1

print("Porcentagem de acertos distância euclidiana:", 100 * acertos_eu / len(teste))
print("Porcentagem de acertos distância manhattan:", 100 * acertos_man / len(teste))
print("Porcentagem de acertos distância mink:", 100 * acertos_mink / len(teste))
print("Porcentagem de acertos distância che:", 100 * acertos_che / len(teste))
