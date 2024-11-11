import numpy as np
import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()
Y_COLUMN = "category_desc"
def train_test_split(X, y, test_size=0.2, random_seed=1):
    # Converte y para DataFrame, se necessário
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y, columns=[Y_COLUMN])

    data = pd.concat([X, y], axis=1)
    # Embaralha os dados
    data = data.sample(frac=1, random_state=random_seed)
    X = data.drop(Y_COLUMN, axis=1)
    y = data[[Y_COLUMN]]  # Transforma y em uma série para que seja 1D
    # Implementação manual do split
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_seed=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print("Teste: ", y_test["category_desc"].value_counts())
print("Treino: ", y_train["category_desc"].value_counts())
