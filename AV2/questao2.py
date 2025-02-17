import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

file_path = "healthcare-dataset-stroke-data-classification.csv"
df = pd.read_csv(file_path)
df.drop(columns=['id'], inplace=True)

# Tratar valores nulos na coluna 'bmi' substituindo pela média
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Codificar variáveis categóricas
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Remover a coluna alvo ('stroke') para análise de clusters
df.drop(columns=['stroke'], inplace=True)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Determinar a quantidade ideal de clusters pelo Método do Cotovelo
wcss = []  # Soma dos quadrados intra-cluster
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot do método do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Método do Cotovelo')
plt.grid()
plt.show()

# Avaliação pelo Método da Silhueta
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot do método da Silhueta
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='red')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Índice de Silhueta')
plt.title('Método da Silhueta')
plt.grid()
plt.show()

# Exibir o melhor número de clusters
best_k = k_values[np.argmax(silhouette_scores)]
print(f"O número ideal de clusters, baseado na Silhueta, é: {best_k}")
