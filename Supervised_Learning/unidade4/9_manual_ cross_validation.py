import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression

class KFold:

   def __init__(self,n_splits):

       self.n_splits = n_splits

   def _compute_score(self,obj,X,y):
       obj.fit(X[0], y[0])
       return obj.score(X[1], y[1])

   def cross_val_score(self,obj,X, y):

        # parte 1: dividir o dataset X em n_splits vezes

        # parte 2: Calcular a métrica score para subset dividida na parte 1. Chamar a função _compute_score para cada subset

        #appendar na lista scores cada valor obtido na parte 2

        #parte 3 - retornar a lista de scores

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


sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object
kf = KFold(n_splits=6)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = kf.cross_val_score(reg,X, y)

# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))

