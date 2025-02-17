import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Carregar os dados e remover a coluna alvo
df = pd.read_csv('wineqt_ajustado.csv')
X = df.drop(columns=['quality'])

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Método do Cotovelo
inertia = []
for k in range(1, 11):  # Testando de 1 a 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotando o gráfico do Cotovelo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.show()

# 2. Método da Silhueta
silhouette_avg = []
for k in range(2, 11):  # Testando de 2 a 10 clusters (pois para k=1 a silhueta não é definida)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    silhouette_avg.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotando o gráfico da Silhueta
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_avg, marker='o')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente de Silhueta')
plt.show()
