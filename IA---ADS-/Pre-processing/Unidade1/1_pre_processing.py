from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
#print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
#print(volunteer.info)

#mostre quantos elementos do dataset estão faltando na coluna
#print(volunteer.locality)

# Exclua as colunas Latitude e Longitude de volunteer
#volunteer_cols = volunteer.drop(colums = ['Latitude', 'Longitude'])
#print(volunteer_cols.head())

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
#volunteer_subset = volunteer_cols.dropna(subset='categoria_desc')
#print(volunteer_subset.head())

# Print o shape do subset
#print(volunteer_subset.shape)


