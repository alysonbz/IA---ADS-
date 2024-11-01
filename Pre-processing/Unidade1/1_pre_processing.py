from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("Dimensão")
print(volunteer.shape)
print("\n")

#mostre os tipos de dados existentes no dataset
print("Tipos")
print(volunteer.dtypes)
print("\n")

#mostre quantos elementos do dataset estão faltando na coluna
print("Quantos estao faltando na coluna")
print(volunteer.isna().sum())
print("\n")

# Exclua as colunas Latitude e Longitude de volunteer
print("Removendo Longitude e Latitude")
volunteer_cols = volunteer.drop(columns="Longitude")
volunteer_cols = volunteer_cols.drop(columns="Latitude")
print(volunteer_cols.isna().sum())
print("\n")

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
print("Removendo os Nulo do Categoty_desc")
volunteer_subset = volunteer_cols.dropna(subset=["category_desc"])
print(volunteer_subset.isna().sum())
print("\n")

# Print o shape do subset
print("Shape do subset")
print(volunteer_subset.shape)


