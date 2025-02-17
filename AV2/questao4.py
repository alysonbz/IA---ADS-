import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Carregar o dataset
file_path = "healthcare-dataset-stroke-data-classification.csv"
df = pd.read_csv(file_path)

# Remover a coluna ID
df.drop(columns=['id'], inplace=True)

# Tratar valores nulos na coluna 'bmi' substituindo pela média
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Codificar variáveis categóricas
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Separar variáveis independentes e dependente
X = df.drop(columns=['stroke'])  # Variáveis para clustering
y = df['stroke']  # Alvo para comparação

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir o melhor K da questão anterior (obtido pelo índice de Silhueta)
best_k_silhouette = 2  # Substitua pelo valor real encontrado na questão 3

# Aplicar K-Means com esse K
kmeans = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Criar a tabela cruzada
crosstab_result = pd.crosstab(df['cluster'], df['stroke'], normalize='index') * 100

# Exibir resultados
print("\nDistribuição de Clusters por Classe-Alvo (Stroke):")
print(crosstab_result.round(2))

# Visualizar melhor os resultados
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.heatmap(crosstab_result, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.xlabel("Classe Stroke (0 = Não, 1 = Sim)")
plt.ylabel("Clusters")
plt.title("Distribuição dos Clusters por Classe-Alvo")
plt.show()
