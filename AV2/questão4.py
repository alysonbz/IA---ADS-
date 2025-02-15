from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

# Separar features e target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir o número ideal de clusters a partir do índice de silhueta
best_k_silhouette = 2

# Melhor número de clusters de acordo com o metodo da silhueta
best_k = range(2, 11)[np.argmax(best_k_silhouette)]
print(f"Melhor número de clusters segundo a silhueta: {best_k}")

# Aplicando KMeans com o melhor número de clusters
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Criando o crosstab para verificar a distribuição dos clusters em relação ao Outcome
crosstab_result = pd.crosstab(df['Cluster'], df['Outcome'])
print("Crosstab entre Clusters e Outcome:")
print(crosstab_result)

# Criando o gráfico para visualização da distribuição de clusters em relação ao Outcome
crosstab_result.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#4CAF50', '#FF9800'], width=0.8)

# Adicionando título e rótulos
plt.title('Distribuição dos Clusters em Relação ao Outcome', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Número de Observações', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Outcome', loc='upper right')

# Exibindo o gráfico
plt.tight_layout()
plt.show()
