import matplotlib.pyplot as plt
from src.utils import load_new_dataframe_gender_classification
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

gender_df = load_new_dataframe_gender_classification()

inertias = []
silhouette_scores = []

ks = range(1, 11)

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(gender_df)
    inertias.append(model.inertia_)

# Plotar o gráfico do metodo do cotovelo
plt.figure(figsize=(8, 6))
plt.plot(ks, inertias, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inertia')
plt.show()

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(gender_df)
    score = silhouette_score(gender_df, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar o gráfico do índice de silhueta
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Índice de Silhueta para Diferentes Valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()