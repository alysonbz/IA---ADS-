from AV1.questao1 import load_water_quality
import pandas as pd
import numpy as np

waterQuality = load_water_quality()

def knn(treinamento_eu, nova_amostra, K):
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
