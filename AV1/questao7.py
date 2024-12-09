import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Funções para calcular as métricas
def compute_RSS(predictions, y):
    return np.sum(np.square(y - predictions))

def compute_MSE(predictions, y):
    return np.mean(np.square(y - predictions))

def compute_RMSE(predictions, y):
    return np.sqrt(compute_MSE(predictions, y))

def compute_R_squared(predictions, y):
    total_variance = np.sum(np.square(y - np.mean(y)))
    explained_variance = np.sum(np.square(predictions - np.mean(y)))
    return explained_variance / total_variance

# Implementação do KFold manual
class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def cross_val_score(self, model, X, y):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_size = n_samples // self.n_splits
        scores = []

        for i in range(self.n_splits):
            # Criar os indices de teste e treino
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            # Dividir em conjuntos de treino e teste
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model.fit(X_train, y_train)

            # Fazer previsões no conjunto de teste
            y_pred = model.predict(X_test)

            # Calcular as métricas para este fold
            RSS = compute_RSS(y_pred, y_test)
            MSE = compute_MSE(y_pred, y_test)
            RMSE = compute_RMSE(y_pred, y_test)
            R_squared = compute_R_squared(y_pred, y_test)

            # Armazenar as métricas
            scores.append((RSS, MSE, RMSE, R_squared))

        return scores


merged_data = pd.read_csv("dataset/regression/merged_ferrari_tesla.csv")
X = merged_data[["High"]].values
y = merged_data["Close"].values
kf = KFold(n_splits=10)

# Criar o modelo de regressão linear
regressor = LinearRegression()
cv_scores = kf.cross_val_score(regressor, X, y)

# métricas para cada fold
for fold, (RSS, MSE, RMSE, R_squared) in enumerate(cv_scores):
    print(f"Fold {fold + 1} -  RSS: {RSS:.2f};  MSE: {MSE:.2f};  RMSE: {RMSE:.2f}; R²: {R_squared:.4f}")
    print("-" * 40)

# exibir as métricas médias para todos os folds
mean_RSS = np.mean([score[0] for score in cv_scores])
mean_MSE = np.mean([score[1] for score in cv_scores])
mean_RMSE = np.mean([score[2] for score in cv_scores])
mean_R_squared = np.mean([score[3] for score in cv_scores])

print("Média das Métricas (em todos os folds):")
print(f"  Média RSS: {mean_RSS:.2f}")
print(f"  Média MSE: {mean_MSE:.2f}")
print(f"  Média RMSE: {mean_RMSE:.2f}")
print(f"  Média R²: {mean_R_squared:.4f}")
