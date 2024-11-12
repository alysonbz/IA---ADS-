import numpy as np

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size):
    treino = np.random.rand(len(X)) < test_size

    X_train = X[treino]
    X_test = X[~treino]
    y_train = y[treino]
    y_test = y[~treino]

    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=["category_desc"])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop("category_desc", axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer['category_desc']

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train.value_counts(), '\n')
print(y_test.value_counts())