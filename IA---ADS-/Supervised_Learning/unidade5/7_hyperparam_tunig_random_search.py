
import numpy as np

#import Logistic regression
____

#import train_test_split
from sklearn.model_selection import train_test_split

#import kfold
___

# Import RandomizedSearchCV
___

from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



#inicialize Logistic regression
logreg  = __

#inicialize kfold
kf =__

#create the parameter space
params = {"penalty": ["____", "____"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(____, ____, ____),
         "class_weight": ["____", {0:____, 1:____}]}

# Instantiate the RandomizedSearchCV object
logreg_cv = ____(____, ____, cv=____)

# Fit the data to the model
logreg_cv.____(____, ____)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(____.____))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(____.____))