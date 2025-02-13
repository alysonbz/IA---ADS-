from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from src.utils import load_new_dataframe_gender_classification
from sklearn.model_selection import train_test_split

#Carregar os dados
gender = load_new_dataframe_gender_classification()

X = gender.drop(['gender'], axis=1)
y = gender['gender'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = KNeighborsClassifier()

#Definir hiperparâmetros a testar
parametros = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Valores para K
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Diferentes métricas de distância
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(modelo, parametros, cv=kf, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

#Resultados
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor acurácia média:", grid_search.best_score_)

# 8. Testar no conjunto de teste
melhor_modelo = grid_search.best_estimator_
acuracia_teste = melhor_modelo.score(X_test, y_test)
print("Acurácia no conjunto de teste:", acuracia_teste)
