from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrgido sem nenhum NAN, removam as colunas NAN e depois as linhas e crie
#um dataframe novo.
print(volunteer.isna().sum())
volunteer_meio = volunteer.drop(['is_priority','amsl','amsl_unit','primary_loc','Community Board','Borough','NTA','Census Tract','BIN','BBL','Latitude','Longitude'], axis=1)
print(volunteer_meio.isna().sum())
volunteer_final = volunteer_meio.dropna()
print(volunteer_final.info())