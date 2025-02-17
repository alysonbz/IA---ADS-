from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

# Carregar os dados
df = pd.read_csv('dataset/wineqt_ajustado.csv')

# Separar as variáveis independentes (features) e a variável dependente (target)
X = df.drop(columns=['quality'])
y = df['quality']

# Normalizar os dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Seleção dos dois atributos mais relevantes (como feito anteriormente com LassoCV)
# (Vamos assumir que a variável `top_2_features` já foi definida, com os dois melhores atributos)
top_2_features = ['alcohol', 'volatile acidity']  # Substitua pelos atributos selecionados
X_selected = X[top_2_features]

# Ajustando o KMeans com o valor de k obtido pelo índice de silhueta
k_silhueta = 2
kmeans = KMeans(n_clusters=k_silhueta, random_state=42)

# Previsões dos clusters
y_pred = kmeans.fit_predict(X_selected)

# Criar o crosstab para analisar a distribuição dos clusters por classe alvo (quality)
cluster_distribution = pd.crosstab(y, y_pred, rownames=['Quality'], colnames=['Cluster'])

# Exibir o crosstab
print(cluster_distribution)
