import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Carregar o dataset (substitua pelo caminho correto se necessário)
df = pd.read_csv("diabetes.csv")

# Separar features e target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir os hiperparâmetros para busca
grid_params = {
    'n_neighbors': list(range(1, 21)),
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance']
}

# Criar o modelo KNN
knn = KNeighborsClassifier()

# Configurar o GridSearchCV
grid_search = GridSearchCV(knn, grid_params, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

# Melhor configuração encontrada
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Avaliação do melhor modelo
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Exibir resultados
print("Melhores parâmetros:", best_params)
print("Acurácia no conjunto de teste:", accuracy)
print("Relatório de classificação:\n", report)

