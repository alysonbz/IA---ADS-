import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
    'broad bean\norigin'
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

# Selecionar apenas features numéricas
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
X_numeric = df[numeric_features].values

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Treinar o K-Means com k=3 (obtido pelo coeficiente de silhueta)
k_silhouette = 3  # Valor de k obtido na Questão 3
kmeans = KMeans(n_clusters=k_silhouette, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Adicionar os rótulos dos clusters ao DataFrame
df['cluster'] = labels_kmeans

# Verificar a coluna alvo (rating)
if 'rating' in df.columns:
    # Garantir que a coluna 'rating' seja numérica
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['rating'], inplace=True)  # Remover valores NaN na coluna 'rating'

    # Criar classes discretas para o rating (ex.: baixo, médio, alto)
    bins = [0, 2.5, 3.5, 5]  # Definir intervalos para as classes
    labels = ['baixo', 'médio', 'alto']
    df['rating_class'] = pd.cut(df['rating'], bins=bins, labels=labels)

    # Gerar o crosstab entre clusters e classes de rating
    crosstab = pd.crosstab(df['cluster'], df['rating_class'])
    print("\nDistribuição dos Clusters em Relação às Classes de Rating:")
    print(crosstab)

    # Plotar o crosstab como um heatmap
    plt.figure(figsize=(8, 5))
    plt.title("Distribuição dos Clusters por Classes de Rating")
    plt.xlabel("Classe de Rating")
    plt.ylabel("Cluster")
    plt.xticks(rotation=45)
    plt.imshow(crosstab, cmap='Blues', aspect='auto')
    plt.colorbar(label='Contagem')
    plt.show()
else:
    print("\nA coluna 'rating' não está presente no dataset.")
