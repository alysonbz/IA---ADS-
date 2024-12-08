import pandas as pd

# Carregar o dataset
WineQT = pd.read_csv('./dataset/WineQT.csv')

# Verificar valores ausentes
print(WineQT.isna().sum())

# Calcular correlações com a coluna 'quality'
correlacoes = WineQT.corr()["quality"].sort_values(ascending=False)

# Filtrar colunas relevantes (correlação absoluta > 0.2)
relevantes = correlacoes[correlacoes.abs() > 0.2]

# Gerar automaticamente o DataFrame com as colunas relevantes
colunas_relevantes = ["quality"] + list(relevantes.index[1:])  # Inclui 'quality'
new_wineqt = WineQT[colunas_relevantes]

# Exibir o DataFrame final
print("\nDataFrame Final:")
print(new_wineqt.head())

# Mostrar a distribuição de classes
print("\nDistribuição de Classes:")
print(new_wineqt["quality"].value_counts())

new_wineqt.to_csv('./dataset/wineqt_ajustado.csv', index=False)
print("As alterações foram salvas no arquivo 'wineqt_ajustado.csv'.")