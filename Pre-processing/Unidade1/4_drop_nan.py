from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.

print("Antes de remover os NAN")
print(volunteer.isna().sum())
print("---------------------------------------")

volunteer_columns = volunteer.dropna(axis=1, how='any')

volunteer_rows = volunteer_columns.dropna(axis=0, how='any')

volunteer_new = volunteer_rows

print("Depois de remover os NAN")
print(volunteer_new.isna().sum())
print("---------------------------------------")
print(volunteer_new.shape)