import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from dataset.BD import load_creditcard
from sklearn.model_selection import train_test_split, GridSearchCV


df = load_creditcard()
dfe = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(n=30000, random_state=42))
df.drop(["id"], axis=1)
X = dfe.drop(["Class"], axis=1)
y = dfe["Class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_params = {
    'n_neighbors': np.arange(1, 51, 2),
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()

# Executa o GridSearchCV
grid_search = GridSearchCV(knn, grid_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor configuração encontrada
best_params = grid_search.best_params_
best_knn = grid_search.best_estimator_

# Avaliação no conjunto de teste
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Melhores parâmetros:", best_params)
print("Acurácia no conjunto de teste:", accuracy)