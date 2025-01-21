from src.utils import load_points
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carregar os pontos
points = load_points()

# Criar uma instância do KMeans com 3 clusters
model = KMeans(n_clusters=3)

# Separar os pontos de teste e treino
test_points = points[:50, :]  # primeiros 50 pontos
train_points = points[50:, :]  # restantes

# Ajustar o modelo aos pontos de treinamento
model.fit(train_points)

# Prever os rótulos dos pontos de teste
labels = model.predict(test_points)

# Imprimir os rótulos dos clusters dos pontos de teste
print(labels)

# Atribuir as colunas de test_points às variáveis xs e ys
xs = test_points[:, 0]
ys = test_points[:, 1]

# Criar um scatter plot dos pontos de teste com as cores baseadas nos rótulos
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Obter os centróides dos clusters
centroids = model.cluster_centers_

# Atribuir as colunas dos centróides às variáveis centroids_x e centroids_y
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Fazer um scatter plot dos centróides
plt.scatter(centroids_x, centroids_y, s=50, marker='D', color='red')

plt.show()
