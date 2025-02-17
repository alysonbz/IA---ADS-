from Avaliações.AV1.src.utils import load_water_quality
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

water_df = load_water_quality()

ks = range(1, 11)
inertias = []
silhouette = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(water_df)
    inertias.append(model.inertia_)

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(water_df)
    silhouette.append(silhouette_score(water_df, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(ks, inertias, marker='o')
axes[0].set_title('Método do Cotovelo')
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].grid()

axes[1].plot(range(2, 11), silhouette, marker='o')
axes[1].set_title('Índice de Silhueta')
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid()

plt.tight_layout()

plt.show()
