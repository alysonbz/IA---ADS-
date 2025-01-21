import numpy as np
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import KFold
from sklearn.model_selection import KFold
# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Import classification_report para avaliar as métricas
from sklearn.metrics import classification_report

from src.utils import load_diabetes_clean_dataset

# Carregar o dataset
diabetes_df = load_diabetes_clean_dataset()

# Dividir as variáveis independentes e dependentes
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o modelo LogisticRegression
logreg = LogisticRegression(solver='liblinear')

# Inicializar KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Criar o espaço de parâmetros
params = {
    "penalty": ["l1", "l2"],  # Tipos de regularização
    "tol": np.linspace(0.0001, 1.0, 50),  # Tolerância para o critério de parada
    "C": np.linspace(0.001, 10, 50),  # Parâmetro de regularização
    "class_weight": ["balanced", None]  # Pesos de classes para lidar com desequilíbrio
}

# Instanciar o RandomizedSearchCV
logreg_cv = RandomizedSearchCV(estimator=logreg, param_distributions=params, cv=kf, n_iter=10, scoring='accuracy', random_state=42)

# Ajustar os dados ao modelo
logreg_cv.fit(X_train, y_train)

# Imprimir os melhores parâmetros e a melhor pontuação
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score on CV: {}".format(logreg_cv.best_score_))

# Avaliar o modelo no conjunto de teste
y_pred = logreg_cv.predict(X_test)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))
