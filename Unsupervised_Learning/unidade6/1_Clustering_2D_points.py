import sys
import os
import matplotlib
matplotlib.use('Agg')  #
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(src_path)

# Tentar importar load_points
try:
    from utils import load_points
except ModuleNotFoundError:
    raise ModuleNotFoundError()

# Carregar os pontos
points = load_points()

# Criar uma instância do KMeans com 3 clusters: model
model = KMeans(n_clusters=3, random_state=42)

# Dividir os pontos em teste e treino
test_points = points[:50, :]
train_points = points[50:, :]

# Ajustar o modelo aos pontos de treinamento
model.fit(train_points)

# Determinar os rótulos dos clusters para os novos pontos: labels
labels = model.predict(test_points)

# Imprimir os rótulos dos clusters
print(labels)

# Atribuir as colunas de test_points para xs e ys
xs = test_points[:, 0]
ys = test_points[:, 1]

# Fazer um scatter plot de xs e ys, usando labels para definir as cores
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Atribuir os centróides dos clusters
centroids = model.cluster_centers_

# Atribuir as colunas dos centróides
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Fazer um scatter plot dos centróides
plt.scatter(centroids_x, centroids_y, s=50, marker='D', color='red')

# Salvar o gráfico em um arquivo
plt.savefig('grafico_clusters.png')
print("Gráfico salvo como 'grafico_clusters.png'.")
