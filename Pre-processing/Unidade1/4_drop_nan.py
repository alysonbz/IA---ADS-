from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, removam as colunas NAN e depois as linhas e crie
#um dataframe novo
volunteer_void = volunteer.drop(['is_priority', 'amsl', 'amsl_unit', 'primary_loc'], axis=1)

print(volunteer_void.isna().sum())

final_volunteer = volunteer_void.dropna()

print(final_volunteer.isna().sum())