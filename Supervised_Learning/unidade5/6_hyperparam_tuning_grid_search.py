import numpy as np

#import Lasso
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#import train_test_split
from sklearn.model_selection import train_test_split

#import kfold
from sklearn.model_selection import KFold

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['glucose'],axis=1)
y = diabetes_df['glucose'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#inicialize Lasso
lasso  = Lasso()
ridg = Ridge()

#inicialize kfold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)
ridg_cv = GridSearchCV(ridg, param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)
ridg_cv.fit(X_train, y_train)

print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))
print("\n")
print("Tuned ridg paramaters: {}".format(ridg_cv.best_params_))
print("Tuned ridg score: {}".format(ridg_cv.best_score_))
