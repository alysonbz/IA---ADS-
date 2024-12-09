from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


# Funções para cálculo de métricas
def compute_RSS(predictions, y):
    return np.sum(np.square(y - predictions))


def compute_MSE(predictions, y):
    return compute_RSS(predictions, y) / len(predictions)


def compute_RMSE(predictions, y):
    return np.sqrt(compute_MSE(predictions, y))


def compute_R_squared(predictions, y):
    total_variance = np.sum(np.square(y - np.mean(y)))
    residual_variance = compute_RSS(predictions, y)
    return 1 - (residual_variance / total_variance)


# Classe KFold customizada
class CustomKFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def cross_val_score(self, model, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)

        return scores


# Carregar o dataset
Regression = pd.read_csv('./dataset/Regression_ajustado.csv')

# Preparar os dados
X = Regression["sp500 open"].values.reshape(-1, 1)
y = Regression["sp500 low"].values

# Instanciar o modelo de regressão linear
reg = LinearRegression()

# Treinar o modelo com todos os dados
reg.fit(X, y)
pred = reg.predict(X)

# Aplicar validação cruzada
kf = CustomKFold(n_splits=6)
cv_score = kf.cross_val_score(reg, X, y)

# Exibir resultados
print("Cv Score: ", cv_score)
print("Média: ", np.mean(cv_score))
print("Desvio Padrão: ", np.std(cv_score))
print("\n")
print("RSS: {}".format(compute_RSS(pred, y)))
print("MSE: {}".format(compute_MSE(pred, y)))
print("RMSE: {}".format(compute_RMSE(pred, y)))
print("R_Squared2: {}".format(compute_R_squared(pred, y)))
