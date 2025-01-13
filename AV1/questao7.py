from AV1.src.utils import load_car_price_prediction
from sklearn.linear_model import LinearRegression
import numpy as np

carPrice = load_car_price_prediction()

def compute_RSS(predictions, y):
    aux = np.square(y - predictions)
    RSS = np.sum(aux)
    return RSS

def compute_MSE(predictions, y):
    RSS = compute_RSS(predictions, y)
    MSE = np.divide(RSS, len(predictions))
    return MSE

def compute_RMSE(predictions, y):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE

def compute_R_squared(predictions, y):
    var_pred = np.sum(np.square(y - np.mean(y)))
    var_data = compute_RSS(predictions, y)
    r_squared = np.divide(var_pred, var_data)
    return r_squared

class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, obj, X, y):
        obj.fit(X[0], y[0])
        return obj.score(X[1], y[1])

    def cross_val_score(self, reg, X, y):
        scores = []
        n_samples = len(X)
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples

            X_test = X[start:end]
            y_test = y[start:end]
            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])

            score = self._compute_score(reg, (X_train, X_test), (y_train, y_test))
            scores.append(score)

        return scores

X = carPrice["Gear box type"].values.reshape(-1, 1)
y = carPrice["Price"].values

kf = KFold(n_splits=6)

reg = LinearRegression()
reg.fit(X, y)

pred = reg.predict(X)

cv_score = kf.cross_val_score(reg, X, y)

print("Cross_validation Score: ", cv_score)
print("Média: ", np.mean(cv_score))
print("Desvio padrão: ", np.std(cv_score))
print(f"RSS: {compute_RSS(pred, y)}")
print(f"MSE: {compute_MSE(pred, y)}")
print(f"RMSE: {compute_RMSE(pred, y)}")
print(f"R^2: {compute_R_squared(pred, y)}")
