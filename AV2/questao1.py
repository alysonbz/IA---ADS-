import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


dataframe = pd.read_csv('../AV1/datasets/train_ajustado.csv')
X = dataframe.drop(['price_range'], axis=1).values
y = dataframe['price_range'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

param_grid = {
    'n_neighbors': np.arange(1, 21),
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("Melhores parâmetros encontrados:", best_params)
print(f"Acurácia no conjunto de validação cruzada: {best_score * 100:.2f}%")
print(f"Acurácia no conjunto de teste:{test_accuracy * 100:.2f}%", )


results = pd.DataFrame(grid_search.cv_results_)

best_metric = best_params['metric']
subset = results[results['param_metric'] == best_metric]

best_k = best_params['n_neighbors']
best_accuracy = subset[subset['param_n_neighbors'] == best_k]['mean_test_score'].values[0]

plt.figure(figsize=(8, 5))
plt.plot(subset['param_n_neighbors'], subset['mean_test_score'], marker='o', linestyle='-', color='b', label="Acurácia")

plt.axvline(x=best_k, linestyle="--", color="red", label=f"Melhor K = {best_k}")

plt.scatter(best_k, best_accuracy, color="red", zorder=3)

plt.xlabel("Número de Vizinhos (K)")
plt.ylabel("Acurácia")
plt.title(f"Acurácia do KNN para diferentes valores de K (distância {best_metric.capitalize()})")
plt.legend()
plt.grid(True)
plt.show()

