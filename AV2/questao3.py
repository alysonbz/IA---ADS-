import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.metrics import silhouette_score

# Carregar o dataset
file_path = "healthcare-dataset-stroke-data-classification.csv"  # Atualize o caminho conforme necessário
df = pd.read_csv(file_path)

# Remover a coluna ID
df.drop(columns=['id'], inplace=True)

# Tratar valores nulos na coluna 'bmi' substituindo pela média
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Codificar variáveis categóricas
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Separar variáveis independentes e dependente
X = df.drop(columns=['stroke'])
y = df['stroke']  # Alvo apenas para o Lasso

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar Lasso para seleção de atributos
lasso = Lasso(alpha=0.01)  # Alpha pequeno para evitar eliminar todas as variáveis
lasso.fit(X_scaled, y)

# Selecionar os dois atributos mais importantes
importance = np.abs(lasso.coef_)
top_2_indices = importance.argsort()[-2:]  # Pega os dois atributos mais importantes
top_2_features = X.columns[top_2_indices]

print(f"Atributos mais relevantes segundo Lasso: {top_2_features.tolist()}")

# Criar novo dataset apenas com os dois atributos selecionados
X_selected = X_scaled[:, top_2_indices]

# Método do Cotovelo e Silhueta com os atributos selecionados
wcss = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_selected)

    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_selected, labels))

# Gráfico do Método do Cotovelo
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Método do Cotovelo (Atributos Selecionados)')
plt.grid()
plt.show()

# Gráfico do Método da Silhueta
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='red')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Índice de Silhueta')
plt.title('Método da Silhueta (Atributos Selecionados)')
plt.grid()
plt.show()

# Determinar os melhores K de cada método
best_k_elbow = k_values[np.argmin(np.diff(wcss))]
best_k_silhouette = k_values[np.argmax(silhouette_scores)]

print(f"Melhor K pelo Método do Cotovelo: {best_k_elbow}")
print(f"Melhor K pelo Método da Silhueta: {best_k_silhouette}")

# Scatterplots para visualização dos clusters
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Clusterização com K do Cotovelo
kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42, n_init=10).fit(X_selected)
axes[0].scatter(X_selected[:, 0], X_selected[:, 1], c=kmeans_elbow.labels_, cmap='viridis', alpha=0.6)
axes[0].scatter(kmeans_elbow.cluster_centers_[:, 0], kmeans_elbow.cluster_centers_[:, 1], s=100, c='red',
                label='Centroides')
axes[0].set_title(f"Clusters com K={best_k_elbow} (Cotovelo)")
axes[0].set_xlabel(top_2_features[0])
axes[0].set_ylabel(top_2_features[1])
axes[0].legend()

# Clusterização com K da Silhueta
kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10).fit(X_selected)
axes[1].scatter(X_selected[:, 0], X_selected[:, 1], c=kmeans_silhouette.labels_, cmap='viridis', alpha=0.6)
axes[1].scatter(kmeans_silhouette.cluster_centers_[:, 0], kmeans_silhouette.cluster_centers_[:, 1], s=100, c='red',
                label='Centroides')
axes[1].set_title(f"Clusters com K={best_k_silhouette} (Silhueta)")
axes[1].set_xlabel(top_2_features[0])
axes[1].set_ylabel(top_2_features[1])
axes[1].legend()

x = np.random.normal(5.0, 1.0, 1000)
y = np.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y, c='blue', alpha=0.7)

plt.show()
