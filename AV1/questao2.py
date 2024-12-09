import math

# Carregar o dataset
dataset = []

with open("Cancer_Data_Adjusted.csv", "r") as f:
    next(f)  #ignora a 1º linha pois a 1ª contém o cabeçalho e não os dados numéricos
    for linha in f.readlines():
        valores = linha.strip().split(',')
        try:
            # Converter valores em float
            valores = [float(valor) for valor in valores]
            dataset.append(valores)
        except ValueError as e:
            print(f"Erro ao processar linha: {linha}. Erro: {e}")


# Função para contar classes no dataset
def count_classes(data):
    classes = {}
    for item in data:
        classe = item[-1]
        if classe not in classes:
            classes[classe] = 0
        classes[classe] += 1
    return classes


# Funções de cálculo de distâncias
def dist_euclidean(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1) - 1)))


def dist_manhattan(v1, v2):
    return sum(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))


def dist_chebyshev(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))


def dist_mahalanobis(v1, v2, cov_matrix_inv):
    diff = [v1[i] - v2[i] for i in range(len(v1) - 1)]
    return math.sqrt(
        sum(diff[i] * sum(cov_matrix_inv[i][j] * diff[j] for j in range(len(diff))) for i in range(len(diff))))


# Função para calcular distâncias
def calculate_distance(v1, v2, metric="euclidean", cov_matrix_inv=None):
    if metric == "euclidean":
        return dist_euclidean(v1, v2)
    elif metric == "manhattan":
        return dist_manhattan(v1, v2)
    elif metric == "chebyshev":
        return dist_chebyshev(v1, v2)
    elif metric == "mahalanobis" and cov_matrix_inv is not None:
        return dist_mahalanobis(v1, v2, cov_matrix_inv)
    else:
        raise ValueError("Invalid distance metric or missing covariance matrix.")


# Função KNN
def knn(train, sample, K, metric="euclidean", cov_matrix_inv=None):
    distances = {}
    for i, train_sample in enumerate(train):
        dist = calculate_distance(train_sample, sample, metric=metric, cov_matrix_inv=cov_matrix_inv)
        distances[i] = dist

    # Selecionar os K vizinhos mais próximos
    k_nearest = sorted(distances, key=distances.get)[:K]

    # Contar as classes dos vizinhos
    class_count = {}
    for idx in k_nearest:
        label = train[idx][-1]
        class_count[label] = class_count.get(label, 0) + 1

    # Retornar a classe mais comum
    return max(class_count, key=class_count.get)


# Separar dataset em treino e teste
def split_dataset(data, train_ratio=0.6):
    class_counts = count_classes(data)
    train, test = [], []
    current_count = {key: 0 for key in class_counts.keys()}
    max_train = {key: int(train_ratio * count) for key, count in class_counts.items()}

    for item in data:
        label = item[-1]
        if current_count[label] < max_train[label]:
            train.append(item)
            current_count[label] += 1
        else:
            test.append(item)

    return train, test


# Preparar os dados
train, test = split_dataset(dataset)
K = 7  # Número de vizinhos
distance_metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]

# Matriz de covariância para Mahalanobis
cov_matrix = [[0] * len(train[0][:-1]) for _ in range(len(train[0][:-1]))]
cov_matrix_inv = None

if "mahalanobis" in distance_metrics:
    try:
        cov_matrix = [[sum(train[i][j] * train[i][k] for i in range(len(train))) for k in range(len(train[0][:-1]))] for
                      j in range(len(train[0][:-1]))]
        cov_matrix_inv = [[0 if i != j else 1 / cov_matrix[i][i] for j in range(len(cov_matrix))] for i in
                          range(len(cov_matrix))]
    except Exception as e:
        print("Erro ao calcular a matriz de covariância para Mahalanobis:", e)

# Avaliar cada métrica de distância
for metric in distance_metrics:
    correct_predictions = 0
    for sample in test:
        predicted_class = knn(train, sample, K, metric=metric, cov_matrix_inv=cov_matrix_inv)
        if predicted_class == sample[-1]:
            correct_predictions += 1

    accuracy = 100 * correct_predictions / len(test)
    print(f"Acurácia com distância {metric}: {accuracy:.2f}%")
