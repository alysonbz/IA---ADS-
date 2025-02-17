import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
df = pd.read_csv('Cancer_Data.csv')

# Remover colunas irrelevantes
df.drop(columns=["Unnamed: 32", "id"], errors='ignore', inplace=True)

# Separar as colunas de características (excluindo a coluna alvo)
X = df.drop(columns=['diagnosis'])

# Remover colunas constantes
X = X.loc[:, X.std() > 0]

# Normalizar os dados
X_scaled = StandardScaler().fit_transform(X)

# Verificar NaNs ou Infinitos
if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
    print("Valores inválidos encontrados, corrigindo...")
    X_scaled = np.nan_to_num(X_scaled)

# Método do Cotovelo
inertia = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(1, 11)]

# Plotando o gráfico do Método do Cotovelo
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

# Método da Silhueta
silhouette_scores = [silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)) for k in range(2, 11)]

# Plotando o gráfico do Método da Silhueta
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Pontuação da Silhueta')
plt.show()

# Imprimir o número ideal de clusters baseado na análise
ideal_k_cotovelo = np.argmin(np.diff(inertia)) + 2
ideal_k_silhueta = np.argmax(silhouette_scores) + 2

print(f"Cotovelo: {ideal_k_cotovelo}, Silhueta: {ideal_k_silhueta}")
