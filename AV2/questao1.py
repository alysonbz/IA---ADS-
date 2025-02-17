import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Carregar o dataset
file_path = "healthcare-dataset-stroke-data-classification.csv"
df = pd.read_csv(file_path)
df.drop(columns=['id'], inplace=True)

# Tratar valores nulos na coluna 'bmi' substituindo pela média
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Codificar variáveis categóricas
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Separar features e target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir parâmetros para GridSearchCV
param_grid = {
    'n_neighbors': range(1, 20, 2),  # Testa valores ímpares de K
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance']
}

# Aplicar GridSearchCV
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred = best_model.predict(X_test)

# Avaliação do modelo
report = classification_report(y_test, y_pred)

# Exibir resultados
print("Melhores parâmetros encontrados:", best_params)
print("\nRelatório de classificação:\n", report)

acuracia_teste = best_model.score(X_test, y_test)
print("Melhor acurácia média:", grid_search.best_score_)
print("Acurácia no conjunto de teste:", acuracia_teste)