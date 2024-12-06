import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from AV1.src.utils import load_car_price_prediction

# Carregar o dataset
carPrice = load_car_price_prediction()

carPrice = carPrice.drop(['ID','Doors','Levy', 'Model', 'Wheel', 'Color'], axis=1)
carPrice['Engine volume'] = carPrice['Engine volume'].str.replace('Turbo','')
carPrice['Engine volume'] = pd.to_numeric(carPrice['Engine volume'])
carPrice['Mileage'] = carPrice['Mileage'].str.split(" ",expand=True)[0]
carPrice['Mileage'] = pd.to_numeric(carPrice['Mileage'])

# Exibir uma visão geral do dataset
print("Primeiras linhas do dataset:")
print(carPrice.head())

# Identificar colunas numéricas e exibir correlações
corr_matrix = carPrice.corr()
print("\nMatriz de Correlação:")
print(corr_matrix)

# Passo 1: Escolher o alvo para regressão
# Para isso, identificamos uma coluna com valores contínuos (não binária)
target_candidates = [col for col in carPrice.columns if carPrice[col].dtype in [np.float64, np.int64] and len(carPrice[col].unique()) > 10]
print("\nCandidatos para alvo (regressão):", target_candidates)

# Escolher o atributo com base na análise prévia ou por inspeção
target = "Price"  # Substitua por outra variável, caso necessário

# Passo 2: Analisar relevância de atributos
# Ordenar os atributos pela correlação com o alvo
correlations = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
print(f"\nCorrelação com o alvo ({target}):")
print(correlations)

# Atributo mais relevante
most_relevant = correlations.idxmax()
print(f"\nAtributo mais relevante para prever '{target}': {most_relevant}")

# Passo 3: Visualizar graficamente a relação do atributo mais relevante com o alvo
plt.figure(figsize=(8, 6))
sns.scatterplot(x=carPrice[most_relevant], y=carPrice[target], alpha=0.7)
plt.title(f'Relação entre {most_relevant} e {target}')
plt.xlabel(most_relevant)
plt.ylabel(target)
plt.grid()
plt.show()

'''Color, ID'''
