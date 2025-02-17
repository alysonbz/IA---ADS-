import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Carregar o dataset
df = pd.read_csv('Cancer_Data.csv')

# Remover colunas irrelevantes
df.drop(columns=["Unnamed: 32", "id"], inplace=True)

# Separar as colunas de características e a variável alvo
X = df.drop(columns=['diagnosis'])
y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Transformando em binário

# Normalizar os dados
X_scaled = StandardScaler().fit_transform(X)

# Aplicar o método Lasso para selecionar os dois atributos mais relevantes
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

# Selecionar os dois atributos mais importantes
top_features_idx = np.argsort(np.abs(lasso.coef_))[-2:]  # Pegando os 2 maiores coeficientes
top_features = X.columns[top_features_idx]

print(f"Atributos mais relevantes: {top_features.tolist()}")

# Reduzir o dataset para apenas esses dois atributos
X_selected = X_scaled[:, top_features_idx]

# Método do Cotovelo
inertia = [KMeans(n_clusters=k, random_state=42).fit(X_selected).inertia_ for k in range(1, 11)]

# Método da Silhueta
silhouette_scores = [silhouette_score(X_selected, KMeans(n_clusters=k, random_state=42).fit_predict(X_selected)) for k in range(2, 11)]

# Determinar o número ideal de clusters
ideal_k_cotovelo = np.argmin(np.diff(inertia)) + 2
ideal_k_silhueta = np.argmax(silhouette_scores) + 2

print(f"Baseado no Método do Cotovelo, o número ideal de clusters é: {ideal_k_cotovelo}")
print(f"Baseado no Método da Silhueta, o número ideal de clusters é: {ideal_k_silhueta}")

# Plotando a curva do Método do Cotovelo
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.grid(True)

# Plotando a pontuação de Silhueta
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title("Método da Silhueta")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Pontuação de Silhueta")
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotar os clusters se os valores forem diferentes
if ideal_k_cotovelo != ideal_k_silhueta:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatterplot para K do cotovelo
    labels_cotovelo = KMeans(n_clusters=ideal_k_cotovelo, random_state=42).fit_predict(X_selected)
    axes[0].scatter(X_selected[:, 0], X_selected[:, 1], c=labels_cotovelo, cmap='viridis')
    axes[0].set_title(f"K-Means com K={ideal_k_cotovelo} (Cotovelo)")

    # Scatterplot para K da silhueta
    labels_silhueta = KMeans(n_clusters=ideal_k_silhueta, random_state=42).fit_predict(X_selected)
    axes[1].scatter(X_selected[:, 0], X_selected[:, 1], c=labels_silhueta, cmap='viridis')
    axes[1].set_title(f"K-Means com K={ideal_k_silhueta} (Silhueta)")

    plt.show()

# Gerando gráfico para visualização do Clustering separadamente
# Aplicar K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_selected)
centroids = kmeans.cluster_centers_

# Criar o scatterplot com melhoria na visualização
plt.figure(figsize=(8, 6))

# Plotando os pontos dos clusters
scatter = plt.scatter(X_selected[:, 0], X_selected[:, 1], c=labels, cmap='viridis', edgecolors='k', alpha=0.75)

# Adicionando os centróides
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centróides')

plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title(f"K-Means Clustering com K={2}")
plt.legend()
plt.grid(True)
plt.show()
