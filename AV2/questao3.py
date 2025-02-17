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

lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, np.random.rand(X_scaled.shape[0]))
important_features = np.argsort(np.abs(lasso.coef_))[-2:]
X_selected = X.iloc[:, important_features]

# Re-normalização dos atributos selecionados
X_selected_scaled = scaler.fit_transform(X_selected)

# Metodo do Cotovelo
wcss = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_selected_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.title('Método do Cotovelo')
plt.show()

# Metodo da Silhueta
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_selected_scaled)
    silhouette_scores.append(silhouette_score(X_selected_scaled, labels))

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Método da Silhueta')
plt.show()

# Determinar os melhores valores de k para cada metodo
best_k_elbow = k_values[np.argmin(np.diff(wcss))]
best_k_silhouette = k_values[np.argmax(silhouette_scores)]

print(f'Melhor k pelo Método do Cotovelo: {best_k_elbow}')
print(f'Melhor k pelo Método da Silhueta: {best_k_silhouette}')

# Criar scatterplots para os diferentes valores de k
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42, n_init=10).fit(X_selected_scaled)
axes[0].scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=kmeans_elbow.labels_, cmap='viridis', alpha=0.5)
axes[0].set_title(f'Clusters pelo Método do Cotovelo (k={best_k_elbow})')

kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10).fit(X_selected_scaled)
axes[1].scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=kmeans_silhouette.labels_, cmap='viridis', alpha=0.5)
axes[1].set_title(f'Clusters pelo Método da Silhueta (k={best_k_silhouette})')

plt.show()
