import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()

X = diabetes_df.drop(['glucose'], axis=1)
y = diabetes_df['glucose'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lasso = Lasso()


kf = KFold(n_splits=5, shuffle=True, random_state=42)


param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

lasso_cv = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=kf)


lasso_cv.fit(X_train, y_train)


print("Melhores parâmetros do Lasso: {}".format(lasso_cv.best_params_))
print("Melhor score do Lasso: {}".format(lasso_cv.best_score_))
