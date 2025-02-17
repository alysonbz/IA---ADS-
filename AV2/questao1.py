import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Carregar o dataset
dataset = pd.read_csv('Cancer_Data.csv')

# Remover colunas irrelevantes e valores NaN
dataset.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)
dataset.dropna(inplace=True)

# Separar variáveis independentes e alvo
X = dataset.drop(columns=['diagnosis'])
y = dataset['diagnosis'].map({'B': 0, 'M': 1})  # Benigno = 0, Maligno = 1

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verificar se há valores NaN após a transformação
print("Existem valores NaN?", np.isnan(X_scaled).sum())

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Verifica e remove qualquer valor não numérico ou NaN apenas das colunas numéricas
df = dataset.copy()
df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).apply(pd.to_numeric, errors='coerce')

# Verifica novamente se ainda existem NaN
print("Existem valores NaN?", df.isna().sum().sum())

# Definir os parâmetros para busca no GridSearchCV
param_grid = {
    'n_neighbors': list(range(1, 21)),  # Testando k de 1 a 20
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Criar o modelo KNN
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor configuração encontrada
best_params = grid_search.best_params_
print(f"\nMelhor valor de K: {best_params['n_neighbors']}")
print(f"Melhor métrica de distância: {best_params['metric']}")

# Avaliação no conjunto de teste
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# Exibir métricas de acurácia e relatório de classificação
print(f"\nAcurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# Análise gráfica da escolha de k
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
sns.lineplot(x=results['param_n_neighbors'], y=results['mean_test_score'], marker='o')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia Média')
plt.title('Efeito do Número de Vizinhos no Desempenho do KNN')
plt.grid(True)
plt.show()
