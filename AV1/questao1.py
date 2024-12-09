# 1) Neste primeiro exercício você deve realizar manipulação em um dataset com a biblioteca pandas e realizar o pré-processamento deste.
import pandas as pd

from src.utils import Hotel_Reservations

Hoteis = Hotel_Reservations()

# Verificar a dimensão, tipo de dados do dataset
#print("Dimensão do dataset:", Hoteis.shape)

#print("\nTipos de dados de cada coluna:")
#print(Hoteis.dtypes)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.(obs: sem valores null)

#print(Hoteis.info())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe com somente as colunas necesárias.

Hotel_Atualizado = Hoteis.drop(columns=["Booking_ID", "type_of_meal_plan", "required_car_parking_space",
                                        "arrival_year", "arrival_date", 'market_segment_type'], axis=1)

print("Dimensão do dataset atualizado:", Hotel_Atualizado.shape)
print(Hotel_Atualizado.dtypes)


# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário.

# Usando o Pandas para One-Hot Encoding - passar colunas para bolleano
Hotel_Atualizado = pd.get_dummies(Hotel_Atualizado,
                                  columns=['room_type_reserved', 'booking_status'],
                                  drop_first=True)


# DataFrame após a transformação
print(Hotel_Atualizado.head())
print(Hotel_Atualizado.dtypes)

# Salve o dataset atualizado se houver modificações.
Hotel_Atualizado.to_csv('Hotel_Atualizado.csv', index=False)

