import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from dataset.BD import load_creditcard

df = load_creditcard()
df = df.drop(columns=['Class'])
X = df.sample(n=30000, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Seleção de características importantes com Lasso
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, np.random.rand(X_scaled.shape[0]))
important_features = np.argsort(np.abs(lasso.coef_))[-2:]
X_selected = X.iloc[:, important_features]

# Re-normalização dos atributos selecionados
X_selected_scaled = scaler.fit_transform(X_selected)

# Metodo do Cotovelo
wcss = []
k_values = np.arange(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_selected_scaled)
    wcss.append(kmeans.inertia_)

wcss_diffs = np.diff(wcss)
wcss_diffs2 = np.diff(wcss_diffs)
best_k_elbow = k_values[np.argmin(wcss_diffs2)]

# Metodo da Silhueta
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_selected_scaled)
    silhouette_scores.append(silhouette_score(X_selected_scaled, labels))

best_k_silhouette = k_values[np.argmax(silhouette_scores)]

# Exibir resultados
print(f'Melhor k pelo Método do Cotovelo: {best_k_elbow}')
print(f'Melhor k pelo Método da Silhueta: {best_k_silhouette}')

# Plot dos gráficos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(k_values, wcss, marker='o')
axes[0].set_xlabel('Número de Clusters')
axes[0].set_ylabel('WCSS')
axes[0].set_title('Método do Cotovelo')
axes[0].axvline(best_k_elbow, linestyle='--', color='red', label=f'k={best_k_elbow}')
axes[0].legend()

axes[1].plot(k_values, silhouette_scores, marker='o')
axes[1].set_xlabel('Número de Clusters')
axes[1].set_ylabel('Coeficiente de Silhueta')
axes[1].set_title('Método da Silhueta')
axes[1].axvline(best_k_silhouette, linestyle='--', color='red', label=f'k={best_k_silhouette}')
axes[1].legend()

plt.show()

# Criar scatterplots para os diferentes valores de k
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42, n_init=10).fit(X_selected_scaled)
axes[0].scatter(X_selected_scaled[:, 0], X_selected_scaled[:, 1], c=kmeans_elbow.labels_, cmap='viridis', alpha=0.5)
axes[0].set_title(f'Clusters pelo Método do Cotovelo (k={best_k_elbow})')

kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10).fit(X_selected_scaled)
axes[1].scatter(X_selected_scaled[:, 0], X_selected_scaled[:, 1], c=kmeans_silhouette.labels_, cmap='viridis', alpha=0.5)
axes[1].set_title(f'Clusters pelo Método da Silhueta (k={best_k_silhouette})')

plt.show()
