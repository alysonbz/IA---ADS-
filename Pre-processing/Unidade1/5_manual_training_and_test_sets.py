from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size,random_seed=1):
    #SEU CÓDIGO AQUI
    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = __

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = ___

# mostre o balanceamento das classes em 'category_desc'
print(___['category_desc'].__,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.__(__, axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = __[['__']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = ---
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size,random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
___