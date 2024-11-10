from src.utils import load_volunteer_dataset
import numpy as np
from collections import Counter

volunteer = load_volunteer_dataset()

def train_test_split(df, cl_alvo, ):
    X = df.drop(cl_alvo, axis=1)
    y = df[[cl_alvo]]

    msk = np.random.rand(len(df)) < 0.8
    X_train = X[msk]
    X_test = X[~msk]
    y_train = y[msk]
    y_test = y[~msk]

    return X_train, X_test, y_train, y_test

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset='category_desc')

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# # Utiliza a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(volunteer, 'category_desc')

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train.value_counts(), '\n')
print(y_test.value_counts())
