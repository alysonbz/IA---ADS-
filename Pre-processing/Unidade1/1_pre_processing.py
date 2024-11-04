from src.utils import load_hiking_dataset , load_volunteer_dataset
import pandas as pd


volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
'''
# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.info())


#mostre quantos elementos do dataset estão faltando na coluna

print(volunteer.isna().sum())
'''
# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.columns.drop("Latitude", "Longitude")
print(volunteer_cols)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna()

# Print o shape do subset
print(volunteer_subset.shape)


