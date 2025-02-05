import numpy as np

#import Logistic regression
from sklearn.linear_model import LogisticRegression

#import train_test_split
from sklearn.model_selection import train_test_split

#import kfold
from sklearn.model_selection import KFold

# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#inicialize Logistic regression
logreg  = LogisticRegression()

#inicialize kfold
kf = KFold(5, shuffle=True, random_state=42)

#create the parameter space
params = {"penalty": ["l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit the data to the model
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))