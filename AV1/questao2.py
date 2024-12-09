import numpy as np
import pandas as pd
import math

# dataset
data_path = '../AV1/dataset/Hotel_Atualizado.csv'
dataset = pd.read_csv(data_path)
print("Dimensão do dataset:", dataset.shape)
print(dataset.info())

# Amostragem aleatória para reduzir o tamanho do dataset (testar com 10%)
tamanho_amostra = 0.1
dataset_reduzido = dataset.sample(frac=tamanho_amostra, random_state=42)

# Separar o dataset em X (features) e y (classe/target)
X = dataset_reduzido.drop(columns=["booking_status_Not_Canceled"]).values
y = dataset_reduzido["booking_status_Not_Canceled"].values

# Divisão manual em treino e teste
def dividir_dataset(X, y, proporcao_treino=0.7):
    tamanho_treino = int(len(X) * proporcao_treino)
    indices = list(range(len(X)))
    np.random.shuffle(indices)  # Embaralhar os índices

    indices_treino = indices[:tamanho_treino]
    indices_teste = indices[tamanho_treino:]

    X_treino = X[indices_treino]
    y_treino = y[indices_treino]
    X_teste = X[indices_teste]
    y_teste = y[indices_teste]

    return X_treino, y_treino, X_teste, y_teste


# Dividir os dados em treino e teste
X_train, y_train, X_test, y_test = dividir_dataset(X, y, proporcao_treino=0.7)

print(f"Tamanho do conjunto de treino: {len(X_train)}")
print(f"Tamanho do conjunto de teste: {len(X_test)}")


# Funções de distância
def dist_euclidiana(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def dist_manhattan(v1, v2):
    return sum(abs(a - b) for a, b in zip(v1, v2))


def dist_chebyshev(v1, v2):
    return max(abs(a - b) for a, b in zip(v1, v2))


def dist_mahalanobis(v1, v2, VI):
    delta = np.array(v1) - np.array(v2)
    return np.sqrt(delta @ VI @ delta.T)


X_train = np.array(X_train, dtype=float)
cov_matrix = np.cov(X_train.T)
VI = np.linalg.pinv(cov_matrix)


# Implementar o KNN
def knn_manual(X_train, y_train, nova_amostra, k, dist_func):
    distancias = []
    for i, amostra_treino in enumerate(X_train):
        if dist_func == dist_mahalanobis:
            distancia = dist_func(amostra_treino, nova_amostra, VI)
        else:
            distancia = dist_func(amostra_treino, nova_amostra)
        distancias.append((distancia, y_train[i]))

    # Ordenar por distância e pegar os K vizinhos mais próximos
    distancias = sorted(distancias, key=lambda x: x[0])
    k_vizinhos = distancias[:k]

    # Contar as classes dos vizinhos
    votos = {}
    for _, classe in k_vizinhos:
        votos[classe] = votos.get(classe, 0) + 1

    # Retornar a classe com mais votos
    return max(votos, key=votos.get)


# Avaliação do KNN
def avaliar_knn(X_train, y_train, X_test, y_test, k, dist_func):
    acertos = 0
    for i, amostra in enumerate(X_test):
        predicao = knn_manual(X_train, y_train, amostra, k, dist_func)
        if predicao == y_test[i]:
            acertos += 1

        # Exibir o progresso de 100 em 100 amostras para acompanhar a apuação
        #if i % 100 == 0:
            #print(f"Progresso: Amostra {i} - Predição: {predicao}, Verdadeiro: {y_test[i]}")
    acuracia = acertos / len(X_test)
    print(f"Acurácia final: {acuracia * 100:.2f}%")
    return acuracia


# Comparação de distâncias
k = 7
distancias = {
    "Euclidiana": dist_euclidiana,
    "Manhattan": dist_manhattan,
    "Chebyshev": dist_chebyshev,
    "Mahalanobis": dist_mahalanobis
}

resultados = {}
for nome, funcao in distancias.items():
    precisao = avaliar_knn(X_train, y_train, X_test, y_test, k, funcao)
    resultados[nome] = precisao * 100
    print(f"Precisão usando {nome}: {precisao * 100:.2f}%")

# Exibir resultados
print("\nResultados Finais:")
for distancia, precisao in resultados.items():
    print(f"{distancia}: {precisao:.2f}%")
