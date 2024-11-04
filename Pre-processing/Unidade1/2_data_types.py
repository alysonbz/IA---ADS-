from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer.head())

# Print as caracteristicas da coluna hits
print(volunteer.info)

# Converta a coluna hits para o tipo int

volunteer_new = volunteer.astype("int64")

# Print as caracteristicas da coluna hits novamente
print(volunteer_new.info)
