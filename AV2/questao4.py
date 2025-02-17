import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

dataframe = pd.read_csv('../AV1/datasets/train_ajustado.csv')
X = dataframe.drop(columns=['price_range']).values
y_true = dataframe['price_range'].values

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

crosstab_result = pd.crosstab(y_true,clusters)

print("\n===== Crosstab - Distribuição dos Clusters por Classe Verdadeira =====")
print(crosstab_result)

plt.figure(figsize=(8, 5))
sns.heatmap(crosstab_result, annot=True, fmt='d', cmap='Blues')
plt.title("Distribuição dos Clusters (K=2) vs Classes Verdadeiras")
plt.xlabel("Cluster Predito")
plt.ylabel("Classe Verdadeira")
plt.show()
