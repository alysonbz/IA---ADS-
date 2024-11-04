from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrgido sem nenhum NAN, removam as colunas NAN e depois as linhas e crie
#um dataframe novo.

print(volunteer.info())

volunteer_cols = volunteer.drop(['is_priority', 'amsl', 'amsl_unit', 'primary_loc'], axis=1)

volunteer_final = volunteer_cols.dropna()

print(volunteer_final.info())
