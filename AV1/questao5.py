import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
Regression = pd.read_csv('./dataset/financial_regression.csv')

# Exibir as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(Regression.head())

# Excluindo linhas com NaN
regression = Regression.dropna(axis=0)

regression_drop_date = regression.drop("date", axis=1)

# Calcular a matriz de correlação
corr_matrix = regression_drop_date.corr()
print("\nMatriz de Correlação:")
print(corr_matrix)

# Selecionar possíveis candidatos a variáveis alvo para regressão
target_candidates = [
    col for col in regression_drop_date.columns
    if regression_drop_date[col].dtype in [np.float64, np.int64] and len(regression_drop_date[col].unique()) > 10
]
print("\nCandidatos para alvo (regressão):", target_candidates)

# Definir a variável alvo
target = "sp500 open"  

# Calcular a correlação com o alvo, ordenando por magnitude
correlations = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
print(f"\nCorrelação com o alvo ({target}):")
print(correlations)

# Encontrar o atributo mais relevante
most_relevant = correlations.idxmax()
print(f"\nAtributo mais relevante para prever '{target}': {most_relevant}")

# Visualizar a relação entre a variável mais relevante e o alvo
plt.figure(figsize=(8, 6))
sns.scatterplot(x=regression_drop_date[most_relevant], y=regression_drop_date[target], alpha=0.7)
plt.title(f'Relação entre {most_relevant} e {target}')
plt.xlabel(most_relevant)
plt.ylabel(target)
plt.grid()
plt.show()

regression_drop_date.to_csv('./dataset/Regression_ajustado.csv', index=False)
print("As alterações foram salvas no arquivo 'Regression_ajustado.csv'.")