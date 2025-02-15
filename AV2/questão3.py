from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import numpy as np

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

# Separar features e target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando Lasso para selecionar as características mais importantes
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

# Obtendo os coeficientes absolutos e selecionando os dois atributos mais relevantes
coef_abs = np.abs(lasso.coef_)
top2_indices = np.argsort(coef_abs)[-2:]  # Índices dos dois atributos mais importantes
top2_features = X.columns[top2_indices]

# Criando um gráfico de barras para visualizar a importância das características
plt.figure(figsize=(10, 5))
plt.bar(X.columns, coef_abs, color='blue', alpha=0.7)
plt.xlabel("Atributos")
plt.ylabel("Valor Absoluto do Coeficiente Lasso")
plt.title("Importância dos Atributos segundo Lasso")
plt.xticks(rotation=45)
plt.axhline(y=coef_abs[top2_indices[0]], color='red', linestyle='--', label=f"Top 2: {top2_features[0]}")
plt.axhline(y=coef_abs[top2_indices[1]], color='green', linestyle='--', label=f"Top 2: {top2_features[1]}")
plt.legend()
plt.show()

# Retornando os dois atributos mais relevantes
print(top2_features)

# Criando um novo dataset com apenas esses dois atributos
X_selected = X_scaled[:, top2_indices]

wcss_selected = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_selected)
    wcss_selected.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss_selected, marker='o', linestyle='--')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.title('Método do Cotovelo (Atributos Selecionados)')
plt.show()

silhouette_scores_selected = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_selected)
    silhouette_avg = silhouette_score(X_selected, labels)
    silhouette_scores_selected.append(silhouette_avg)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores_selected, marker='o', linestyle='--')
plt.xlabel('Número de Clusters')
plt.ylabel('Índice de Silhueta')
plt.title('Método da Silhueta (Atributos Selecionados)')
plt.show()

# Melhor número de clusters segundo a silhueta
best_k_silhouette = range(2, 11)[np.argmax(silhouette_scores_selected)]
print(f"Melhor número de clusters segundo a silhueta: {best_k_silhouette}")

# Rodando o KMeans para os valores encontrados
best_k_elbow = 3
kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42, n_init=10)
labels_elbow = kmeans_elbow.fit_predict(X_selected)

kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10)
labels_silhouette = kmeans_silhouette.fit_predict(X_selected)

# Criando os scatterplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=labels_elbow, cmap='viridis', edgecolors='k')
plt.xlabel(top2_features[0])
plt.ylabel(top2_features[1])
plt.title(f'Clusters pelo Método do Cotovelo (k={best_k_elbow})')

plt.subplot(1, 2, 2)
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=labels_silhouette, cmap='plasma', edgecolors='k')
plt.xlabel(top2_features[0])
plt.ylabel(top2_features[1])
plt.title(f'Clusters pelo Método da Silhueta (k={best_k_silhouette})')

plt.show()
