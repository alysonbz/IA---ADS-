from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrgido sem nenhum NAN, removam as colunas NAN e depois as linhas e crie
#um dataframe novo.

print(volunteer.info())

volunteerNaN = volunteer.drop(["is_priority", "amsl", "primary_loc","amsl_unit"], axis=1)

volunteerNaN = volunteerNaN.dropna()

print(volunteerNaN)