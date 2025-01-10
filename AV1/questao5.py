import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o dataset
data = pd.read_csv(r'kc_house_data.csv')

# 2. Verificar valores NaN e removê-los, se houver
data_cleaned = data.dropna()

# Dropandando as colunas date e id
data_cleaned = data.drop(columns=["date", "id"])

# 3. Definir o alvo para regressão
# O alvo é o preço da casa ('price')
target = "price"

# 4. Identificar a correlação das variáveis com o alvo
correlations = data_cleaned.corr()[target].sort_values(ascending=False)
print("Correlação das variáveis com o preço:")
print(correlations)

# 5. Selecionar atributos mais relevantes
# Mantendo apenas colunas com correlação forte ou moderada (threshold arbitrário: |correlation| > 0.2)
relevant_columns = correlations[correlations.abs() > 0.2].index

# Criar novo dataframe com as colunas relevantes
data_relevant = data_cleaned[relevant_columns]

# 6. Visualizar a relação do atributo mais relevante com o preço (além do próprio preço)
most_relevant_feature = correlations.index[1]  # O mais relevante após o próprio preço

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_cleaned[most_relevant_feature], y=data_cleaned[target])
plt.title(f"Relação entre {most_relevant_feature} e {target}")
plt.xlabel(most_relevant_feature)
plt.ylabel(target)
plt.show()

# 7. Salvar o dataset pré-processado
data_relevant.to_csv('king_county_preprocessed.csv', index=True)