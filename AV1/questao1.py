# 1) Neste primeiro exercício você deve realizar manipulação em um dataset com a biblioteca pandas e realizar o pré-processamento deste.
import pandas as pd
from src.utils import Hotel_Reservations
from sklearn.preprocessing import MinMaxScaler, StandardScaler

Hoteis = Hotel_Reservations()

# Verificar a dimensão, tipo de dados do dataset
#print("Dimensão do dataset:", Hoteis.shape)

#print("\nTipos de dados de cada coluna:")
#print(Hoteis.dtypes)

# 1) Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.(obs: sem valores null)

print(Hoteis.info())

# 2) Verifique quais colunas são as mais relevantes e crie um novo dataframe com somente as colunas necesárias.

Hotel_Atualizado = Hoteis.drop(columns=["Booking_ID", "type_of_meal_plan", "required_car_parking_space",
                                        "arrival_year", "arrival_date", 'market_segment_type'], axis=1)

print("Dimensão do dataset atualizado:", Hotel_Atualizado.shape)
#print(Hotel_Atualizado.dtypes)


#3) Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário.

# Usando o Pandas para One-Hot Encoding - VALOR NUMERICO
Hotel_Atualizado = pd.get_dummies(Hotel_Atualizado,
                                  columns=['room_type_reserved', 'booking_status'],
                                  drop_first=True)

# Selecionar apenas colunas numéricas'
colunas_numericas = Hotel_Atualizado.select_dtypes(include=['float64', 'int64']).columns

# Criar uma cópia do dataframe
Hotel_Normalizado = Hotel_Atualizado.copy()

# aplicar o StandardScaler(slide 35)

print(Hotel_Normalizado.head())
print(Hotel_Normalizado.describe())

# Salvar o dataset normalizado
Hotel_Normalizado.to_csv('Hotel_Normalizado.csv', index=False)
