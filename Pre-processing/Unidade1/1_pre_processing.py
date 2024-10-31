from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("O primeiro número é a quantidade de linhas e o segundo é a quantidade de colunas")
print(volunteer.shape)
print("--------------------")

#mostre os tipos de dados existentes no dataset
print("Esses são os tipos de dados do dataset")
print(volunteer.dtypes)
print("--------------------")

#mostre quantos elementos do dataset estão faltando na coluna
print("Quantidade de dados faltando em cada coluna")
print(volunteer.isna().sum())
print("--------------------")

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)
print("Antes da remoção")
print(volunteer.columns)
print("Depois da remoção")
print(volunteer_cols.columns)
print("--------------------")

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset='category_desc')
print("Coluna category_desc com os valores null excluidos")
print(volunteer_subset['category_desc'])
print("Desmonstração de que os valores null foram removidos")
print(volunteer_subset.isna().sum())
print("--------------------")

# Print o shape do subset
print("Essa é a nova quantidade de linhas e colunas")
print(volunteer_subset.shape)


