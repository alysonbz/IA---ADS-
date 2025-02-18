from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd
from AV2.dataset.BD import load_creditcard

df = load_creditcard()

y = df['Class']
df = df.drop(columns=['Class'])
X = df.sample(n=30000, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Seleção de atributos com Lasso
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, np.random.rand(X_scaled.shape[0]))
important_features = np.argsort(np.abs(lasso.coef_))[-2:]
X_selected = X.iloc[:, important_features]

# Re-normalização dos atributos selecionados
X_selected_scaled = scaler.fit_transform(X_selected)

# Aplicar KMeans com o melhor k pelo índice de silhueta
y_sampled = y.sample(n=30000, random_state=42)
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_selected_scaled)
df_clusters = pd.DataFrame({'Cluster': kmeans_final.labels_, 'Class': y_sampled.values})

# Criar crosstab para análise da distribuição dos clusters
crosstab_result = pd.crosstab(df_clusters['Cluster'], df_clusters['Class'])
print("\nDistribuição dos Clusters por Classe:")
print(crosstab_result)
