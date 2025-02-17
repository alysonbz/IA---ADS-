from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregar os dados
df = pd.read_csv('dataset/wineqt_ajustado.csv')

# Separar as variáveis independentes (features) e a variável dependente (target)
X = df.drop(columns=['quality'])
y = df['quality']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando LassoCV para selecionar os melhores atributos
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# Obter os coeficientes do modelo
coef = pd.Series(lasso.coef_, index=X.columns)

# Selecionar os dois atributos mais relevantes (maiores coeficientes absolutos)
top_2_features = coef.abs().nlargest(2).index
X_selected = X[top_2_features]

print(f"Dois atributos mais relevantes selecionados: {top_2_features}")

# ------------------------------------------------------------

# Aplicar o método do cotovelo para encontrar o número ideal de clusters
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_selected)
    inertia.append(kmeans.inertia_)

# Plotar o gráfico de cotovelo
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

# ------------------------------------------------------------

# Método-da-Silhueta
silhouette_scores = []
for k in range(2, 11):  # Para o método da silhueta, o número de clusters deve ser ao menos 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_selected)
    score = silhouette_score(X_selected, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar gráfico da silhueta
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Pontuação da Silhueta')
plt.show()

# ------------------------------------------------------------

# Número de clusters (baseado nos resultados do método)
k_cotovelo = 3
k_silhueta = 2

# Ajustando o modelo KMeans com os valores de k
kmeans_cotovelo = KMeans(n_clusters=k_cotovelo, random_state=42)
kmeans_silhueta = KMeans(n_clusters=k_silhueta, random_state=42)

# Previsões dos clusters
y_pred_cotovelo = kmeans_cotovelo.fit_predict(X_selected)
y_pred_silhueta = kmeans_silhueta.fit_predict(X_selected)

# Scatterplot para o método do cotovelo
plt.figure(figsize=(8, 6))
plt.scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=y_pred_cotovelo, cmap='viridis')
plt.title(f'Clusters (k={k_cotovelo}) - Método do Cotovelo')
plt.xlabel(top_2_features[0])
plt.ylabel(top_2_features[1])
plt.colorbar()
plt.show()

# Scatterplot para o método da silhueta
plt.figure(figsize=(8, 6))
plt.scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=y_pred_silhueta, cmap='viridis')
plt.title(f'Clusters (k={k_silhueta}) - Método da Silhueta')
plt.xlabel(top_2_features[0])
plt.ylabel(top_2_features[1])
plt.colorbar()
plt.show()
