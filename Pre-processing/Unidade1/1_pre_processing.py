from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)


# Mostre as informações do dataset volunteer
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print(volunteer.isna().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_drop = volunteer.drop(["Longitude","Latitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_drop.dropna(subset='category_desc')

# Print o shape do subset
print(volunteer_subset.shape)
