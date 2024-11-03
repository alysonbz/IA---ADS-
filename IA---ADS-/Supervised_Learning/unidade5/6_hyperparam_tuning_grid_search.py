import numpy as np

#import Lasso
_______

#import train_test_split
from sklearn.model_selection import train_test_split

#import kfold
____

# Import GridSearchCV
______

from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['glucose'],axis=1)
y = diabetes_df['glucose'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#inicialize Lasso
lasso  = ____

#inicialize kfold
kf = _____

#Set up the parameter grid
param_grid = {"____": np.linspace(____, __, ___)}

# Instantiate lasso_cv
lasso_cv = ____

# Fit to the training data
___


print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))