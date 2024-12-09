import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'Lenovo Group Limited (2000-2024).csv'
data = pd.read_csv(file_path)

print(data.info())
print(data.head())

data_cleaned = data.dropna()
numerical_data = data_cleaned.select_dtypes(include=['float64', 'int64'])

print("Colunas numéricas disponíveis para análise:")
print(numerical_data.columns)

# Identificar o alvo para regressão
target_column = 'Close'
if target_column not in numerical_data.columns:
    raise ValueError(f"A coluna alvo '{target_column}' não é numérica.")

correlation_matrix = numerical_data.corr()
print(correlation_matrix[target_column].sort_values(ascending=False))

# Plotar mapa de calor da correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Mapa de Calor da Correlação')
plt.show()

relevant_features = correlation_matrix[target_column][correlation_matrix[target_column].abs() > 0.5].index.tolist()

#criar um novo dataset com os atributos relevantes
data_relevant = data_cleaned[relevant_features]

# Plotar gráficos de dispersão para os atributos relevantes
for feature in relevant_features:
    if feature != target_column:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data_relevant, x=feature, y=target_column)
        plt.title(f'Relação entre {feature} e {target_column}')
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plt.show()


data_relevant.to_csv('Lenovo_ajustado.csv', index=False)
