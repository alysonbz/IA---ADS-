import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from AV1.src.utils import load_car_price_prediction

carPrice = load_car_price_prediction()

print("Primeiras linhas do dataset:")
print(carPrice.head())

corr_matrix = carPrice.corr()
print("\nMatriz de Correlação:")
print(corr_matrix)

target_candidates = [col for col in carPrice.columns if carPrice[col].dtype in [np.float64, np.int64] and len(carPrice[col].unique()) > 10]
print("\nCandidatos para alvo (regressão):", target_candidates)

target = "Price"

correlations = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
print(f"\nCorrelação com o alvo ({target}):")
print(correlations)

most_relevant = correlations.idxmax()
print(f"\nAtributo mais relevante para prever '{target}': {most_relevant}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=carPrice[most_relevant], y=carPrice[target], alpha=0.7)
plt.title(f'Relação entre {most_relevant} e {target}')
plt.xlabel(most_relevant)
plt.ylabel(target)
plt.grid()
plt.show()
