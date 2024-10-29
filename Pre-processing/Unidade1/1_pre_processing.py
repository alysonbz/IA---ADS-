from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("Print do tamanho do dataset")
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print("Print das características do dataset")
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print("Elementos faltando na coluna locality")
print(volunteer["locality"].isna().sum())

# Exclua as colunas Latitude e Longitude de volunteer
print("Removendo Lat e Lon do dataset")
volunteer_cols = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
print("Removendo NaNs da col category_desc do dataset")
volunteer_subset = volunteer_cols.dropna(subset="category_desc")
print(volunteer_subset["category_desc"].isna().sum())

# Print o shape do subset
print("shape do dataset volunteer_shape")
print(volunteer_subset.shape)


