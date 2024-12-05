import numpy as np
from src.utils import load_new_dataframe_kc_house
from sklearn.linear_model import LinearRegression

house_df = load_new_dataframe_kc_house()

def compute_RSS(predictions, y):
    # Diferenças quadráticas somadas
    RSS = np.sum(np.square(y - predictions))
    return RSS
def compute_MSE(predictions,y):
    RSS = compute_RSS(predictions, y)
    MSE = np.divide(RSS, len(predictions))
    return MSE
def compute_RMSE(predictions,y):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE
def compute_R_squared(predictions, y):
    ss_total = np.sum(np.square(y - np.mean(y)))  # Soma dos quadrados totais
    ss_residual = compute_RSS(predictions, y)    # Soma dos quadrados residuais
    r_squared = 1 - (ss_residual / ss_total)     # Fórmula do R^2
    return r_squared

class KFold:

   def __init__(self,n_splits):

       self.n_splits = n_splits

   def _compute_score(self,obj,X,y):
       obj.fit(X[0], y[0])
       return obj.score(X[1], y[1])

   def cross_val_score(self,obj,X, y):
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

            score = self._compute_score(obj, (X_train, X_test), (y_train, y_test))
            scores.append(score)

        return scores

# Create X and y arrays
X = house_df["sqft_living"].values.reshape(-1,1)
y = house_df["price"].values

kf = KFold(n_splits=6)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X,y)

predictions = reg.predict(X)

cv_scores = kf.cross_val_score(reg,X, y)

print("Cv Score: ",cv_scores)
print("Média: ",np.mean(cv_scores))
print("Desvio Padrão: ",np.std(cv_scores))
print("\n")
print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))