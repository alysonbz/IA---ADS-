import math

lista=[]
with open('../dataset/iris.data', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        lista.append(a)

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if lista[i][4] == 'Iris-setosa':
            setosa += 1
        if lista[i][4] == 'Iris-versicolor':
            versicolor += 1
        if lista[i][4] == 'Iris-virginica':
            virginica += 1

    return [setosa, versicolor, virginica]

p=0.6
setosa,versicolor, virginica = countclasses(lista)
treinamento, teste= [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1 = 0
total2 = 0
total3 = 0
for lis in lista:
    if lis[-1] == 'Iris-setosa' and total1< max_setosa:
        treinamento.append(lis)
        total1 +=1
    elif lis[-1] == 'Iris-versicolor' and total2<max_versicolor:
        treinamento.append(lis)
        total2 +=1
    elif lis[-1] == 'Iris-virginica' and total3<max_virginica:
        treinamento.append(lis)
        total3 +=1
    else:
        teste.append(lis)

def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(float(v1[i]) - float(v2[i]),2)
    return math.sqrt(soma)
#Distancias
def dist_manhatta(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += abs(float(v1[i]) - float(v2[i]))
    return soma

def dist_minkowski(v1, v2, p_val):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += abs(float(v1[i]) - float(v2[i])) ** p_val
    return soma ** (1 / p_val)

def dist_chebyshev(v1, v2):
    dim = len(v1)
    maximo = 0
    for i in range(dim - 1):
        distancia = abs(float(v1[i]) - float(v2[i]))
        if distancia > maximo:
            maximo = distancia
    return maximo


def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 'Iris-setosa':
            qtd_setosa += 1
        elif treinamento[indice][-1] == 'Iris-versicolor':
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0

def knnManhatta(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_manhatta(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 'Iris-setosa':
            qtd_setosa += 1
        elif treinamento[indice][-1] == 'Iris-versicolor':
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0

def knnMinkowski(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_minkowski(treinamento[i], nova_amostra, 100)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 'Iris-setosa':
            qtd_setosa += 1
        elif treinamento[indice][-1] == 'Iris-versicolor':
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0

def knnChebyshev(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_chebyshev(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 'Iris-setosa':
            qtd_setosa += 1
        elif treinamento[indice][-1] == 'Iris-versicolor':
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    #Adicionei um if e depois comparei com o nome da iris
    if classe == 1:
        if amostra[-1] == 'Iris-setosa':
            acertos +=1
    elif classe == 2:
        if amostra[-1] == 'Iris-versicolor':
            acertos +=1
    elif classe == 3:
        if amostra[-1] == 'Iris-virginica':
            acertos +=1
print("Porcentagem de acertos Euclidianas:" , 100*acertos/len(teste))

#Mahatta
acertos = 0
for amostra in teste:
    classe = knnManhatta(treinamento, amostra, K)
    #Adicionei um if e depois comparei com o nome da iris
    if classe == 1:
        if amostra[-1] == 'Iris-setosa':
            acertos +=1
    elif classe == 2:
        if amostra[-1] == 'Iris-versicolor':
            acertos +=1
    elif classe == 3:
        if amostra[-1] == 'Iris-virginica':
            acertos +=1
print("Porcentagem de acertos Manhatta:" , 100*acertos/len(teste))

#minkowski
acertos = 0
for amostra in teste:
    classe = knnMinkowski(treinamento, amostra, K)
    #Adicionei um if e depois comparei com o nome da iris
    if classe == 1:
        if amostra[-1] == 'Iris-setosa':
            acertos +=1
    elif classe == 2:
        if amostra[-1] == 'Iris-versicolor':
            acertos +=1
    elif classe == 3:
        if amostra[-1] == 'Iris-virginica':
            acertos +=1
print("Porcentagem de acertos Minkowvski:" , 100*acertos/len(teste))

#Chebyshev
acertos = 0
for amostra in teste:
    classe = knnChebyshev(treinamento, amostra, K)
    #Adicionei um if e depois comparei com o nome da iris
    if classe == 1:
        if amostra[-1] == 'Iris-setosa':
            acertos +=1
    elif classe == 2:
        if amostra[-1] == 'Iris-versicolor':
            acertos +=1
    elif classe == 3:
        if amostra[-1] == 'Iris-virginica':
            acertos +=1
print("Porcentagem de acertos Chebyshev:" , 100*acertos/len(teste))