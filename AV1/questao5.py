import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o dataset de regressão
dataset = pd.read_csv('Possum_Data_Adjusted.csv')

# Exibir as primeiras linhas do dataset para inspecionar
print(dataset.head())

# Identificar colunas com valores NaN e removê-las
dataset = dataset.dropna()

# Verificar se há colunas com valores NaN
print(f"Colunas com NaN: \n{dataset.isna().sum()}")

# Definir o alvo (target) para a regressão
target_column = 'age'

# Separar as variáveis independentes (X) e dependentes (y)
X = dataset.drop(columns=[target_column])
y = dataset[target_column]

# Análise de relevância: Matriz de correlação
correlation_matrix = dataset.corr()

# Plotando a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlação")
plt.show()

# Plotando a correlação entre as variáveis independentes e o alvo
plt.figure(figsize=(10, 6))
sns.scatterplot(x=dataset['totlngth'], y=dataset[target_column])
plt.title(f"Correlação entre 'totlngth' e {target_column}")
plt.xlabel('totlngth')
plt.ylabel(target_column)
plt.show()

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização (se necessário)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Visualização final para mostrar a relação entre os atributos e o alvo
plt.figure(figsize=(10, 6))
sns.pairplot(dataset, x_vars=X.columns, y_vars=[target_column], height=3)
plt.title("Distribuição dos Atributos vs Alvo")
plt.show()

