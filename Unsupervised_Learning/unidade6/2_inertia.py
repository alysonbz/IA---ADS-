import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

# Carregar o dataset de grãos
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)

# Definir o intervalo de clusters a ser testado
ks = range(1, 6)
inertias = []

# Loop para calcular a inércia para cada valor de k
for k in ks:
    # Criar uma instância do KMeans com k clusters
    model = KMeans(n_clusters=k, random_state=42)

    # Ajustar o modelo aos dados
    model.fit(samples)

    # Adicionar a inércia do modelo à lista de inércias
    inertias.append(model.inertia_)

# Plotar o número de clusters (k) versus a inércia
plt.plot(ks, inertias, '-o')
plt.xlabel('Número de clusters, k')
plt.ylabel('Inércia')
plt.xticks(ks)
plt.title('Número de Clusters vs Inércia')
plt.show()
