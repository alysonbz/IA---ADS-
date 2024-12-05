import numpy as np

from src.utils import load_volunteer_dataset
import math

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size,random_seed=1):
    def train_test_split(X, y, train_size):
        msk = np.random.rand(len(X)) < train_size
        X_train = X[msk]
        X_test = X[~msk]
        y_train = y[msk]
        y_test = y[~msk]

        return X_train, X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer_new.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer_new['category_desc']

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train.value_counts(),'\n','\n', print(y_test.value_counts()))