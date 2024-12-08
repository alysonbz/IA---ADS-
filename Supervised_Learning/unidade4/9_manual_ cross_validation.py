import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression

def compute_RSS(predictions, y):
    RSS = np.sum(np.square(y - predictions))
    return RSS

def compute_R_squared(predictions, y):
    r_squared = (compute_RSS(predictions, np.mean(y)) / compute_RSS(y, np.mean(y)))
    return r_squared

class KFold:

   def __init__(self,n_splits):

        self.n_splits = n_splits

   def _compute_score(self,y_true,y_pred):
        r_squared = (compute_RSS(y_pred, np.mean(y_true)) / compute_RSS(y_true, np.mean(y_true)))
        return r_squared

   def cross_val_score(self,model,X, y):

        scores = []

        # parte 1: dividir o dataset X em n_splits vezes
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_size = n_samples // self.n_splits
        scores = []

        for i in range(self.n_splits):
            # criar subsets para cada fold
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            # criar train e test sets
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # parte 2: Calcular a métrica score para subset dividida na parte 1. Chamar a função _compute_score para cada subset
            score = self._compute_score(y_test, y_pred)
            # appendar na lista scores cada valor obtido na parte 2
            scores.append(float(score))

        #parte 3 - retornar a lista de scores

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
