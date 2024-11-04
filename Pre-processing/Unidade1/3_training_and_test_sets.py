from src.utils import load_volunteer_dataset
_____

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer_new.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer_new[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 stratify=y, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print("Treino:\n", y_train.value_counts(), '\n')
print("Teste: \n", y_test.value_counts(), '\n')