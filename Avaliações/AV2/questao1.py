from Avaliações.AV1.src.utils import load_water_quality
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

water_df = load_water_quality()

X = water_df.drop(['is_safe'], axis=1)
y = water_df['is_safe'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

modelo = KNeighborsClassifier()

param_grid = {
    'n_neighbors': list(range(1, 21)),
    'metric': ['euclidean', 'manhattan', 'minkowski']
              }

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(modelo, param_grid, cv=kf, scoring='accuracy')
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)
print("Melhor acurácia:", grid.best_score_)
