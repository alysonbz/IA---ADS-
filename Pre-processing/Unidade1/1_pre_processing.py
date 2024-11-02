import pandas as pd

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("dimensão do dataset:", volunteer.shape)

#mostre os tipos de dados existentes no dataset
print("Informações do Dataset")
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print("Elementos que faltam no DATASET")
print(volunteer.isna().sum())

# Exclua as colunas Latitude e Longitude de volunteer
print("Excluindo Colunas Longitude e Latitude")
volunteer_cols = volunteer.drop(columns=["Latitude", "Longitude"], axis=1)


# Exclua as linhas com valores null da coluna category_desc de volunteer_cols

volunteer_subset = volunteer_cols.dropna(subset=["category_desc"], axis=0)



# Print o shape do subset
print("Shape Volunteer-Subset")
print(volunteer_subset.shape)


