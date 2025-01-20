import numpy as np

#import Lasso & Ridge
from sklearn.linear_model import Lasso, Ridge

#import train_test_split
from sklearn.model_selection import train_test_split

#import kfold
from sklearn.model_selection import KFold, cross_val_score

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV


from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['glucose'],axis=1)
y = diabetes_df['glucose'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#inicialize Lasso
lasso = Lasso()


#inicialize kfold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, num=10)}


# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)

print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))

# Inicializar Ridge
ridge = Ridge()

# Definir o grid de parâmetros para Ridge
param_grid_ridge = {
    "alpha": np.linspace(0.0001, 1, num=10),  # Alpha com valores ajustados
    "solver": ["sag", "lsqr"]  # Corrigido o nome do parâmetro
}

# Inicializar GridSearchCV para Ridge
ridge_cv = GridSearchCV(ridge, param_grid_ridge, cv=kf)

# Ajustar modelo Ridge
ridge_cv.fit(X_train, y_train)

# Exibir os melhores parâmetros e o melhor score para Ridge
print("Tuned Ridge parameters: {}".format(ridge_cv.best_params_))
print("Tuned Ridge score: {}".format(ridge_cv.best_score_))