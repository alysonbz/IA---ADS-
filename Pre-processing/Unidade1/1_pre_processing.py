from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.type)
#mostre quantos elementos do dataset estão faltando na coluna
print(volunteer.isnull.sum())
# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop([['Latitude', 'Longitude']])

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer.dropna([['category_desc', 'volunteer_cols']])

# Print o shape do subset
print(volunteer_subset.shape)


