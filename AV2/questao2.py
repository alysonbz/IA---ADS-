import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


dataframe = pd.read_csv('../AV1/datasets/train_ajustado.csv')
X = dataframe.drop(columns=['price_range']).values

k_values = range(2, 11)
inertia = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

#Metodo do Cotovelo(Inercia)
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Definição de K')
plt.grid()
plt.show()

#Silhueta
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, marker='s', linestyle='-', color='g')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Coeficiente de Silhueta para Definição de K')
plt.grid()
plt.show()

best_k = k_values[np.argmax(silhouette_scores)]
print(f"Melhor número de clusters segundo o método da Silhueta: {best_k}")




#Análise mais aprofundada sobre o K (Decisão Pessoal)


k_values = [2, 3, 4]

# PCA para reduzir os dados para 2 dimensões (somente para visualização)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#Criar gráficos para cada valor de K
plt.figure(figsize=(15, 5))

for i, k in enumerate(k_values, 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    centroids_pca = pca.transform(kmeans.cluster_centers_)

    plt.subplot(1, 3, i)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, label="Centróides")
    plt.title(f"K-Means com K={k}")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()

plt.tight_layout()
plt.show()


y_true = dataframe['price_range'].values

for k in k_values:
    print(f"\n===== AVALIAÇÃO PARA K = {k} =====")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_pred), yticklabels=np.unique(y_true))
    plt.xlabel("Clusters Preditos")
    plt.ylabel("Classes Verdadeiras")
    plt.title(f"Matriz de Confusão - K={k}")
    plt.show()

    #Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
