import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

# Carregar o dataset de grãos
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)

# Definir o intervalo de valores para k
ks = range(1, 6)
inertias = []

# Iterar sobre os valores de k
for k in ks:
    # Criar uma instância do KMeans com k clusters
    model = KMeans(n_clusters=k)

    # Ajustar o modelo aos dados
    model.fit(samples)

    # Adicionar a inércia do modelo à lista de inércias
    inertias.append(model.inertia_)

# Plotar os valores de k vs inércia
plt.plot(ks, inertias, '-o')
plt.xlabel('Número de clusters, k')
plt.ylabel('Inércia')
plt.xticks(ks)
plt.title('Método do Cotovelo')
plt.show()
