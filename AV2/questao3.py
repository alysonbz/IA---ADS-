import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Caminho local para o arquivo CSV
file_path = "flavors_of_cacao.csv"

# Carregar o dataset
df = pd.read_csv(file_path)

# Normalizar os nomes das colunas (remover espaços extras e converter para minúsculas)
df.columns = df.columns.str.strip().str.lower()

# Pré-processamento: Remover colunas irrelevantes
columns_to_drop = [
    'company\xa0\n(maker-if known)',
    'specific bean origin\nor bar name',
    'broad bean\norigin',
    'rating'  # Remover a coluna alvo
]
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True)

# Tratar valores nulos (remover linhas com valores ausentes)
df.dropna(inplace=True)

# Converter a coluna 'cocoa percent' para numérica (remover o símbolo '%' e converter para float)
if 'cocoa\npercent' in df.columns:
    df['cocoa\npercent'] = df['cocoa\npercent'].str.replace('%', '', regex=True).astype(float)

# Crição features
if 'cocoa\npercent' in df.columns:
    df['high_cocoa'] = (df['cocoa\npercent'] > 70).astype(int)  # Feature binária para alto teor de cacau
if 'review\ndate' in df.columns:
    df['decade'] = (df['review\ndate'] // 10) * 10  # Extrair a década da data de revisão

# Selecionar apenas features numéricas
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
X_numeric = df[numeric_features].values

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Seleção de Features com Lasso
# Usar uma feature contínua como alvo simulado (ex.: 'cocoa\npercent')
if 'cocoa\npercent' in df.columns:
    y_simulated = df['cocoa\npercent']
else:
    y_simulated = np.random.randint(0, 2, size=len(df))  # Fallback para rótulos aleatórios

lasso = Lasso(alpha=0.01)  # Regularização Lasso
lasso.fit(X_scaled, y_simulated)  # Usar rótulos simulados
feature_importance = pd.Series(lasso.coef_, index=numeric_features)
top_features = feature_importance.abs().nlargest(2).index  # Duas features mais relevantes
print("\nDuas features mais relevantes selecionadas pelo Lasso:")
print(top_features)

# Filtrar apenas as duas features mais relevantes
X_relevant = df[top_features].values

# Normalizar novamente os dados
X_relevant_scaled = scaler.fit_transform(X_relevant)

# Método do Cotovelo
sse = []  # Soma dos quadrados intra-cluster
k_range = range(2, 11)  # Testar de 2 a 10 clusters
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_relevant_scaled)
    sse.append(kmeans.inertia_)  # SSE

# Coeficiente de Silhueta
silhouette_scores = []
for k in range(2, 11):  # Silhueta só faz sentido para k >= 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_relevant_scaled)
    score = silhouette_score(X_relevant_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Comparar com a Questão 2
print("\nComparação com a Questão 2:")
print("Número ideal de clusters na Questão 2: [3] (Método do Cotovelo)")
print("Número ideal de clusters na Questão 2: [2] (Coeficiente de Silhueta)")
print(f"Número ideal de clusters na Questão 3: {k_range[np.argmin(sse)]} (Método do Cotovelo)")
print(f"Número ideal de clusters na Questão 3: {k_range[np.argmax(silhouette_scores)]} (Coeficiente de Silhueta)")

# Scatterplots para Análise Visual
k_elbow = k_range[np.argmin(sse)]  # Valor de k do método do cotovelo
k_silhouette = k_range[np.argmax(silhouette_scores)]  # Valor de k do coeficiente de silhueta

if k_elbow != k_silhouette:
    print("\nGerando scatterplots para análise visual...")

    # Scatterplot para k do método do cotovelo
    kmeans_elbow = KMeans(n_clusters=k_elbow, random_state=42)
    labels_elbow = kmeans_elbow.fit_predict(X_relevant_scaled)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_relevant[:, 0], X_relevant[:, 1], c=labels_elbow, cmap='viridis', s=50)
    plt.title(f'Scatterplot (k={k_elbow} - Método do Cotovelo)')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.colorbar(label='Cluster')
    plt.show()

    # Scatterplot para k do coeficiente de silhueta
    kmeans_silhouette = KMeans(n_clusters=k_silhouette, random_state=42)
    labels_silhouette = kmeans_silhouette.fit_predict(X_relevant_scaled)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_relevant[:, 0], X_relevant[:, 1], c=labels_silhouette, cmap='viridis', s=50)
    plt.title(f'Scatterplot (k={k_silhouette} - Coeficiente de Silhueta)')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.colorbar(label='Cluster')
    plt.show()

    # Análise Visual
    print(f"\nAnálise Visual:")
    print(f"- Para k={k_elbow} (cotovelo), os clusters parecem mais granulares.")
    print(f"- Para k={k_silhouette} (silhueta), os clusters parecem mais compactos e bem separados.")
else:
    print("\nOs métodos concordam sobre o número ideal de clusters. Não é necessário gerar scatterplots.")
