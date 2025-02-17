import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataframe = pd.read_csv('../AV1/datasets/train_ajustado.csv')
X = dataframe.drop(columns=['price_range'])
y = dataframe['price_range']

lasso = Lasso(alpha=0.01)
lasso.fit(X, y)

feature_importance = np.abs(lasso.coef_)
top_features_indices = np.argsort(feature_importance)[-2:]
top_features = X.columns[top_features_indices]
print(f"Atributos mais relevantes segundo Lasso: {list(top_features)}")

X_reduced = X[top_features].values

k_values = range(2, 11)
inertia = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_reduced)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_reduced, kmeans.labels_))

# Metodo do Cotovelo
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para os Dois Atributos Selecionados')
plt.grid()
plt.show()

#Coeficiente de Silhueta
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, marker='s', linestyle='-', color='g')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Coeficiente de Silhueta para os Dois Atributos Selecionados')
plt.grid()
plt.show()

best_k_silhouette = k_values[np.argmax(silhouette_scores)]
best_k_elbow = 3

print(f"Melhor K pelo Método do Cotovelo: {best_k_elbow}")
print(f"Melhor K pelo Coeficiente de Silhueta: {best_k_silhouette}")

if best_k_silhouette != best_k_elbow:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, k in enumerate([best_k_elbow, best_k_silhouette]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_reduced)

        axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200,
                        label="Centróides")
        axes[i].set_title(f"Scatterplot para K={k}")
        axes[i].set_xlabel(top_features[0])
        axes[i].set_ylabel(top_features[1])
        axes[i].legend()

    plt.tight_layout()
    plt.show()
