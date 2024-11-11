from src.utils import load_volunteer_dataset
import random

volunteer = load_volunteer_dataset()

import random


def train_test_split(X, y, test_size=0.2, random_seed=1):
    random.seed(random_seed)

    indices = list(range(len(X)))
    random.shuffle(indices)

    test_size = int(len(X) * test_size)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(["Longitude", "Latitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=["category_desc"])

# Mostre o balanceamento das classes em 'category_desc'
print(volunteer["category_desc"].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de `category_desc`
X = volunteer.drop("category_desc", axis=1)

# Crie um DataFrame de labels com a coluna `category_desc`
y = volunteer[["category_desc"]]

# Utilize a nova função para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=1)

# Mostre o balanceamento das classes em 'category_desc' novamente
print(y_train["category_desc"].value_counts(), '\n', '\n')
print(y_test["category_desc"].value_counts(), '\n', '\n')
