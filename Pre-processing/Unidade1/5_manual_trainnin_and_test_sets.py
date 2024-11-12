import random
import numpy as np
import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

def train_test_split(X, y, test_size=0.2, random_seed=1):
    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Generate random indices
    indices = random.sample(range(len(X)), len(X))
    print("Randomized Indices:", indices)

    # Reorder X and y using the randomized indices
    data_X = X.iloc[indices].reset_index(drop=True)
    data_y = y.iloc[indices].reset_index(drop=True)

    # Calculate the split index
    split_index = int(len(X) * (1 - test_size))

    # Split into training and testing sets
    X_train, X_test = data_X[:split_index], data_X[split_index:]
    y_train, y_test = data_y[:split_index], data_y[split_index:]

    return X_train, X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset="category_desc")

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop("category_desc", axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_seed=40)

# mostre o balanceamento das classes em 'category_desc' novamente
print("Teste: ", y_test["category_desc"].value_counts())
print("Treino: ", y_train["category_desc"].value_counts())
