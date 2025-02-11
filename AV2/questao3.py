from src.utils import load_new_dataframe_gender_classification
from src.utils import load_gender_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

gender = load_new_dataframe_gender_classification()

X = gender.drop(['gender'], axis=1)
y = gender['gender'].values

gender_columns = X.columns

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.1)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(gender_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()

#GRAFICO DE COTOVELO E SILHUETA
gender_gra = gender[['forehead_width_cm','forehead_height_cm']]

inertias = []
silhouette_scores = []

ks = range(1, 11)

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(gender_gra)
    inertias.append(model.inertia_)

# Plotar o gráfico do metodo do cotovelo
plt.figure(figsize=(8, 6))
plt.plot(ks, inertias, marker='o')
plt.title('Metodo do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inertia')
plt.show()

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(gender_gra)
    score = silhouette_score(gender_gra, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar o gráfico do índice de silhueta
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Índice de Silhueta para Diferentes Valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

#SCATTERPLOTS
# 1. Carregar o dataset
df = load_gender_classification()

# 2. Selecionar duas colunas numéricas para agrupar
x_col = "forehead_height_cm"
y_col = "forehead_width_cm"
data = df[[x_col, y_col]]

# 3. Definir número de clusters (K)
k = 4

# 4. Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(data)

# 5. Criar scatter plot com os clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df[x_col],
                y=df[y_col],
                hue=df["cluster"],
                palette="viridis",
                style=df["cluster"],
                alpha=0.7)

# 6. Adicionar os centróides no gráfico
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroides")

# 7. Personalizar o gráfico
plt.title(f"K-Means Clustering (K={k})")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend(title="Cluster")
plt.show()