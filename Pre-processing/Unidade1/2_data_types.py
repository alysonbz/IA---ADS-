from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os 5 primeiros elementos da coluna hits
print(volunteer.head(["hits"]))

# Print as caracteristicas da coluna hits
print(volunteer["hits"].describe())

# Converta a coluna hits para o tipo int
volunteer["hits"] = volunteer["hits"].astype(int)


# Print as caracteristicas da coluna hits novamente
print("Hits inteiro")
print(volunteer["hits"])
