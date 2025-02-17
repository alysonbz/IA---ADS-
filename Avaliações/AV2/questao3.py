from Avaliações.AV1.src.utils import load_water_quality
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

water_df = load_water_quality()

X = water_df.drop(['is_safe'], axis=1)
y = water_df['is_safe'].values

water_columns = X.columns

lasso = Lasso(alpha=0.1)

lasso_coef = lasso.fit(X, y).coef_
plt.figure(figsize=(12, 6))
plt.bar(water_columns, lasso_coef)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Coeficiente do Lasso")
plt.title("Importância das variáveis na qualidade da água")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#GRAFICO DE COTOVELO E SILHUETA
water_gra = water_df[['aluminium', 'chloramine']]

ks = range(1, 11)
inertias = []
silhouette = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(water_gra)
    inertias.append(model.inertia_)

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(water_gra)
    silhouette.append(silhouette_score(water_gra, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(ks, inertias, marker='o')
axes[0].set_title('Método do Cotovelo')
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].grid()

axes[1].plot(range(2, 11), silhouette, marker='o')
axes[1].set_title('Índice de Silhueta para Diferentes Valores de K')
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid()

plt.tight_layout()

plt.show()

#SCATTERPLOT
df = load_water_quality()

x_col = 'aluminium'
y_col = 'chloramine'
data = df[[x_col, y_col]]

k = 3

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(data)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(ax=axes[0], x=df[x_col], y=df[y_col], hue=df["cluster"],
                palette="viridis", style=df["cluster"], alpha=0.7)
centroids = kmeans.cluster_centers_
axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroides")
axes[0].set_title(f"K-Means Clustering (K={k})")
axes[0].set_xlabel(x_col)
axes[0].set_ylabel(y_col)
axes[0].legend(title="Cluster")

k = 5

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(data)
sns.scatterplot(ax=axes[1], x=df[x_col], y=df[y_col], hue=df["cluster"],
                palette="coolwarm", style=df["cluster"], alpha=0.7)
centroids = kmeans.cluster_centers_
axes[1].scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label="Centroides")
axes[1].set_title(f"K-Means Clustering (K={k})")
axes[1].set_xlabel(x_col)
axes[1].set_ylabel(y_col)
axes[1].legend(title="Cluster")

plt.tight_layout()
plt.show()