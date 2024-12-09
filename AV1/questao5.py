import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Obter o caminho do arquivo CSV no mesmo diretório que o script
file_path_regression = os.path.join(os.getcwd(), "Life Expectancy Data.csv")  # O arquivo CSV deve estar no mesmo diretório do script
dataset_regression = pd.read_csv(file_path_regression)

# 2. Remover espaços extras nos nomes das colunas
dataset_regression.columns = dataset_regression.columns.str.strip()

# 3. Verificar as primeiras linhas do dataset
print(dataset_regression.head())

# 4. Analisar os dados, verificar colunas e valores nulos
print(dataset_regression.info())

# 5. Verificar se há colunas com valores NaN
print(dataset_regression.isnull().sum())

# 6. Selecionar o atributo alvo para a regressão (Life expectancy é o alvo)
X_regression = dataset_regression.drop(columns=['Life expectancy'])
y_regression = dataset_regression['Life expectancy']

# 7. Remover colunas não numéricas (Country e Status)
dataset_regression = dataset_regression.select_dtypes(include=[np.number])  # Apenas colunas numéricas

# 8. Tratar valores NaN: preencher colunas numéricas com a média de cada coluna
dataset_regression = dataset_regression.fillna(dataset_regression.mean())

# 9. Verificar novamente os valores ausentes
print(dataset_regression.isnull().sum())

# 10. Verificar as correlações entre as variáveis
correlation = dataset_regression.corr()

# Plotar a matriz de correlação utilizando matplotlib
plt.figure(figsize=(10, 8))
plt.matshow(correlation, fignum=1)
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title('Matriz de Correlação')
plt.show()

# 11. Selecionar as variáveis mais relevantes para a regressão
# Pode-se remover as colunas com baixa correlação com a variável alvo, por exemplo, 'Country'
X_regression = X_regression.drop(columns=['Country'])

# 12. Plotar gráficos de dispersão para explorar a relação das variáveis com o alvo (Life expectancy)
# Usaremos matplotlib para gráficos de dispersão
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.scatter(dataset_regression['GDP'], y_regression)
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.title('GDP vs Life Expectancy')

plt.subplot(2, 3, 2)
plt.scatter(dataset_regression['Alcohol'], y_regression)
plt.xlabel('Alcohol')
plt.ylabel('Life Expectancy')
plt.title('Alcohol vs Life Expectancy')

plt.subplot(2, 3, 3)
plt.scatter(dataset_regression['Schooling'], y_regression)
plt.xlabel('Schooling')
plt.ylabel('Life Expectancy')
plt.title('Schooling vs Life Expectancy')

plt.subplot(2, 3, 4)
plt.scatter(dataset_regression['Income composition of resources'], y_regression)
plt.xlabel('Income composition of resources')
plt.ylabel('Life Expectancy')
plt.title('Income composition vs Life Expectancy')

plt.subplot(2, 3, 5)
plt.scatter(dataset_regression['BMI'], y_regression)
plt.xlabel('BMI')
plt.ylabel('Life Expectancy')
plt.title('BMI vs Life Expectancy')

plt.tight_layout()
plt.show()

# 13. Exibir o dataset final após o pré-processamento
print(X_regression.head())

# 14. Salvar o dataset atualizado
output_file_path = os.path.join(os.getcwd(), 'life_expotancy_data_ajustado.csv')
dataset_regression.to_csv(output_file_path, index=False)
print(f"Arquivo ajustado salvo com sucesso em: {output_file_path}")
