import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

# Criação de features
if 'cocoa\npercent' in df.columns:
    df['high_cocoa'] = (df['cocoa\npercent'] > 70).astype(int)  # Feature binária para alto teor de cacau
if 'review\ndate' in df.columns:
    df['decade'] = (df['review\ndate'] // 10) * 10  # Extrair a década da data de revisão

# Codificar variáveis categóricas
if 'company\nlocation' in df.columns:
    encoder = OneHotEncoder(sparse_output=False)
    location_encoded = encoder.fit_transform(df[['company\nlocation']])
    location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['company\nlocation']))
    df = pd.concat([df, location_df], axis=1)
    df.drop(columns=['company\nlocation'], inplace=True)

if 'bean\ntype' in df.columns:
    encoder = OneHotEncoder(sparse_output=False)
    bean_type_encoded = encoder.fit_transform(df[['bean\ntype']])
    bean_type_df = pd.DataFrame(bean_type_encoded, columns=encoder.get_feature_names_out(['bean\ntype']))
    df = pd.concat([df, bean_type_df], axis=1)
    df.drop(columns=['bean\ntype'], inplace=True)

# Preencher valores nulos nas features com a média
df = df.fillna(df.mean(numeric_only=True))

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Método do Cotovelo
sse = []  # Soma dos quadrados intra-cluster
k_range = range(1, 11)  # Testar de 1 a 10 clusters
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)  # SSE

# Plotar o gráfico do método do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('SSE')
plt.grid()
plt.show()

# Coeficiente de Silhueta
silhouette_scores = []
for k in range(2, 11):  # Silhueta só faz sentido para k >= 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar o gráfico do coeficiente de silhueta
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Coeficiente de Silhueta')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

# Resultados
print("Resultados do Método do Cotovelo:")
print("SSE para cada k:", sse)
print("\nResultados do Coeficiente de Silhueta:")
print("Silhouette Scores para cada k:", silhouette_scores)
