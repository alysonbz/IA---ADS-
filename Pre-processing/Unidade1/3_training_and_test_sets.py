from src.utils import load_volunteer_dataset


volunteer = load_volunteer_dataset()


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.columns.drop("Latitude", "Longitude")

print(volunteer_new)


# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])


# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer_new.drop("category_desc", axis=1)

print(X)


# Crie um dataframe de labels com a coluna category_desc
# y = __[['__']]

'''
# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
___

'''