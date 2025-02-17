import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar o dataset ajustado
df = pd.read_csv('./dataset/wineqt_ajustado.csv')

# Separar features e target
X = df.drop(columns=['quality'])
y = df['quality']

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redução de dimensionalidade
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Divisão entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, random_state=42)

# Definição dos parâmetros para busca
param_grid = {
    'n_neighbors': np.arange(5, 11),
    'metric': ['euclidean', 'manhattan', 'chebyshev'],
    'weights': ['uniform', 'distance']
}

# Definição do modelo KNN
knn = KNeighborsClassifier()

# Configuração do GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Treinar o modelo
grid_search.fit(X_train, y_train)

# Melhor conjunto de parâmetros
best_params = grid_search.best_params_

# Obter o melhor modelo
best_model = grid_search.best_estimator_

# Previsões nos conjuntos de treino e teste
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calcular acurácias
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# Função de avaliação para as métricas de distância
def avaliar_knn_com_metricas(X_train, y_train, X_test, y_test, K, metric):
    knn = KNeighborsClassifier(n_neighbors=K, metric=metric)
    knn.fit(X_train, y_train)

    # Previsões em treino e teste
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    # Calcular acurácias
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    return train_acc, test_acc


# Avaliar o modelo com diferentes métricas de distância
K = best_params['n_neighbors']
print("\nAvaliação detalhada do modelo:")
print(f"Melhores parâmetros encontrados: {best_params}")
print(f"Acurácia no conjunto de treino: {train_accuracy:.4f}")
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")

print(f"\nComparação de métricas de distância com K={K}:")
metrics = ['euclidean', 'manhattan', 'chebyshev']
for metric in metrics:
    train_acc, test_acc = avaliar_knn_com_metricas(X_train, y_train, X_test, y_test, K, metric)
    print(f"\nMétrica: {metric}")
    print(f"- Acurácia treino: {train_acc:.4f}")
    print(f"- Acurácia teste: {test_acc:.4f}")

# Verificar resultados do grid search
print("\nMelhores resultados por configuração:")
cv_results = pd.DataFrame(grid_search.cv_results_)
best_results = cv_results.nlargest(5, 'mean_test_score')
print("\nTop 5 melhores configurações:")
for idx, row in best_results.iterrows():
    params = row['params']
    print(f"\nParâmetros: {params}")
    print(f"Média CV Score: {row['mean_test_score']:.4f}")