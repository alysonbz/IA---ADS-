import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Carregar o dataset
df = pd.read_csv("Cancer_Data.csv")

# Remover colunas desnecessárias
df = df.drop(columns=['Unnamed: 32', 'id'])  # Removendo 'Unnamed: 32' e 'id', que não são relevantes
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Convertendo diagnóstico em 1 (Maligno) e 0 (Benigno)

# Preencher valores ausentes com a média da coluna
df.fillna(df.mean(), inplace=True)

# Separar features e a variável alvo
X = df.drop(columns=['diagnosis'])  # Features: todas as colunas exceto 'diagnosis'
y_real = df['diagnosis']  # Diagnóstico real (0 = Benigno, 1 = Maligno)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inicialização para encontrar o melhor k usando o índice de silhueta
best_k = 2
best_score = -1

# Variáveis para armazenar os resultados para os gráficos
inertia = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(score)

    # Verificar se a pontuação de silhueta é melhor para o k atual
    if score > best_score:
        best_k = k
        best_score = score

# Aplicar K-Means com o melhor k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
y_clusters = kmeans.fit_predict(X_scaled)

# Criar uma crosstab (matriz de confusão) entre diagnóstico real e clusters
crosstab = pd.crosstab(y_real, y_clusters, rownames=['Diagnóstico Real'], colnames=['Cluster'])

# Exibir o melhor valor de k e a crosstab
print(f"Melhor k baseado no índice de silhueta: {best_k}")
print("\nMatriz de Confusão (Crosstab) entre Diagnóstico Real e Clusters:")
print(crosstab)

# Plotar o gráfico do método do cotovelo (Inércia)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Método do Cotovelo (Inércia)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')

# Plotar os scores de silhueta
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Pontuação de Silhueta')

plt.tight_layout()
plt.show()

# Aplicar PCA para reduzir as dimensões para 2 e visualizar os clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotar os clusters após o K-Means com o melhor k (PCA)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_clusters, cmap='viridis')
plt.title(f'K-Means Clustering com k={best_k} - PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()
